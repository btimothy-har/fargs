import asyncio
import json

import ell
import pydantic
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from pydantic import Field
from retry_async import retry

from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import default_retry_config
from fargs.config import default_summarization_llm
from fargs.exceptions import FargsExtractionError
from fargs.exceptions import FargsLLMError
from fargs.models import CommunityReport
from fargs.prompts import COMMUNITY_REPORT
from fargs.utils import async_batch
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent


class CommunitySummarizer(TransformComponent, LLMPipelineComponent):
    prompt: str | None = Field(default=None)
    config: dict = Field(default_factory=dict)

    def __init__(
        self,
        prompt: str = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.prompt = prompt or COMMUNITY_REPORT.format(
            output_schema=CommunityReport.model_json_schema()
        )
        self.config = overwrite_config or default_summarization_llm

        if "gpt-4o" in self.config["model"]:
            self.config["response_format"] = CommunityReport
        else:
            self.config["response_format"] = {"type": "json_object"}

    def _construct_function(self):
        @ell.complex(**self.config)
        def generate_community_report(community_text: str):
            return [
                ell.system(self.prompt),
                ell.user(community_text),
            ]

        return generate_community_report

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        asyncio.run(self.acall(nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        batch_count = 0
        total_batches = (len(nodes) // PROCESSING_BATCH_SIZE) + 1
        transformed = []

        async for batch in async_batch(nodes, batch_size=PROCESSING_BATCH_SIZE):
            batch_count += 1
            tasks = [
                asyncio.create_task(self.summarize_community(node, **kwargs))
                for node in batch
            ]

            async for task in tqdm_iterable(
                tasks,
                f"Batch {batch_count}/{total_batches}: Summarizing communities...",
            ):
                try:
                    result = await task
                    transformed.append(result)
                except Exception as e:
                    print(e)
                    continue

        return transformed

    @retry(
        (FargsLLMError),
        is_async=True,
        **default_retry_config,
    )
    async def summarize_community(self, node: BaseNode, **kwargs) -> BaseNode:
        raw_text = node.text

        raw_result = await self.invoke_llm(community_text=raw_text)

        if "gpt-4o" in self.config["model"]:
            result = raw_result.parsed
        else:
            try:
                raw_report = json.loads(raw_result.text_only)
                result = CommunityReport.model_validate(raw_report)
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                raise FargsExtractionError(
                    f"Failed to validate community report: {e}\n\n"
                    f"{raw_result.text_only}"
                ) from e

        node.text = result.description
        node.metadata["title"] = result.title
        node.metadata["impact_severity_rating"] = result.impact_severity_rating
        node.metadata["rating_explanation"] = result.rating_explanation

        node.excluded_embed_metadata_keys = [
            "entities",
            "impact_severity_rating",
            "sources",
        ]
        node.excluded_llm_metadata_keys = ["entities", "sources"]

        return node
