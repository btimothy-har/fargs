import asyncio
from datetime import UTC
from datetime import datetime
from enum import Enum

from llama_index.core.extractors import BaseExtractor
from retry_async import retry

from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import default_retry_config
from fargs.exceptions import FargsExtractionError
from fargs.models import DefaultClaimTypes
from fargs.models import build_claim_model
from fargs.prompts import EXTRACT_CLAIMS_PROMPT
from fargs.utils import logger
from fargs.utils import sequential_task
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent

CLAIM_EXTRACTION_MESSAGE = """
<entities>
{entities_text}
</entities>

<text>
{text_unit}
</text>
"""


class ClaimsExtractor(BaseExtractor, LLMPipelineComponent):
    def __init__(
        self,
        prompt: str = None,
        claim_types: Enum = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        system_prompt = prompt or EXTRACT_CLAIMS_PROMPT.format(
            current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            claim_types=[t.value for t in self._claim_types],
            output_schema=self._output_model.model_json_schema(),
        )

        component_args = {
            "_component_name": "fargs.claims.extractor",
            "system_prompt": system_prompt,
            "output_model": list[build_claim_model(claim_types or DefaultClaimTypes)],
        }

        if overwrite_config:
            component_args["agent_config"] = overwrite_config

        super().__init__(**component_args, **kwargs)

    async def aextract(self, nodes):
        claims = []

        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]
        async for task in tqdm_iterable(
            tasks,
            "Extracting claims...",
        ):
            try:
                raw_results = await task
                claims.append({"claims": raw_results})
            except Exception as e:
                logger.exception(f"Error extracting claims: {e}")
                claims.append({"claims": None})

        return claims

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    @retry(
        (FargsExtractionError),
        is_async=True,
        **default_retry_config,
    )
    async def invoke_and_parse_results(self, node):
        if not node.metadata.get("entities"):
            return None

        claims = []
        entities_text = "\n\n".join([
            f"NAME: {m.name}, TYPE: {m.entity_type}, DESCRIPTION: {m.description}"
            for m in node.metadata["entities"]
        ])

        claims = await self.invoke_llm(
            user_prompt=CLAIM_EXTRACTION_MESSAGE.format(
                entities_text=entities_text,
                text_unit=node.text,
            )
        )

        return claims
