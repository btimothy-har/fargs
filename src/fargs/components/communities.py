import asyncio

import tiktoken
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from pydantic import PrivateAttr

from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import SUMMARY_CONTEXT_WINDOW
from fargs.models import CommunityReport
from fargs.prompts import COMMUNITY_REPORT
from fargs.utils import logger
from fargs.utils import sequential_task
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent


class CommunitySummarizer(TransformComponent, LLMPipelineComponent):
    _tokenizer: tiktoken.Encoding | None = PrivateAttr(default=None)

    def __init__(
        self,
        prompt: str = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        system_prompt = prompt or COMMUNITY_REPORT

        component_args = {
            "component_name": "fargs.communities.summarizer",
            "system_prompt": system_prompt,
            "output_model": CommunityReport,
        }

        if overwrite_config:
            component_args["agent_config"] = overwrite_config

        super().__init__(**component_args, **kwargs)

        self._tokenizer = (
            tiktoken.get_encoding("o200k_base")
            if "gpt-4o" in self.agent_config["model"]
            else tiktoken.get_encoding("cl100k_base")
        )

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        asyncio.run(self.acall(nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        transformed = []

        tasks = [
            asyncio.create_task(self.summarize_community(node, **kwargs))
            for node in nodes
        ]

        async for task in tqdm_iterable(
            tasks,
            "Summarizing communities...",
        ):
            try:
                result = await task
            except Exception as e:
                logger.exception(f"Error summarizing community: {e}")
                continue
            else:
                transformed.append(result)

        return transformed

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    async def summarize_community(self, node: BaseNode, **kwargs) -> BaseNode:
        text_encoded = await asyncio.to_thread(self._tokenizer.encode, node.text)

        desc = (
            await asyncio.to_thread(
                self._tokenizer.decode, text_encoded[:SUMMARY_CONTEXT_WINDOW]
            )
            if len(text_encoded) > SUMMARY_CONTEXT_WINDOW
            else node.text
        )

        community_report = await self.invoke_llm(user_prompt=desc)

        node.text = community_report.description
        node.metadata["title"] = community_report.title
        node.metadata["impact_severity_rating"] = (
            community_report.impact_severity_rating
        )
        node.metadata["rating_explanation"] = community_report.rating_explanation

        node.excluded_embed_metadata_keys = [
            "entities",
            "impact_severity_rating",
            "sources",
        ]
        node.excluded_llm_metadata_keys = ["entities", "sources"]

        return node
