import asyncio
from datetime import UTC
from datetime import datetime
from enum import Enum

from llama_index.core.extractors import BaseExtractor
from pydantic import BaseModel
from pydantic import Field
from retry_async import retry

from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import LLMConfiguration
from fargs.config import RetryConfig
from fargs.exceptions import FargsExtractionError
from fargs.models import DefaultEntityTypes
from fargs.models import build_entity_model
from fargs.prompts import EXTRACT_ENTITIES_PROMPT
from fargs.utils import logger
from fargs.utils import sequential_task
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent


def build_output_model(base_model: BaseModel) -> BaseModel:
    class EntityOutput(BaseModel):
        entities: list[base_model] = Field(
            title="Entities", description="List of entities identified."
        )
        no_entities: bool = Field(
            title="No Entities Flag",
            description="If there are no entities to identify, set this to True.",
        )

    return EntityOutput


class EntityExtractor(BaseExtractor, LLMPipelineComponent):
    def __init__(
        self,
        prompt: str = None,
        entity_types: Enum = None,
        overwrite_config: LLMConfiguration | None = None,
        **kwargs,
    ):
        system_prompt = prompt or EXTRACT_ENTITIES_PROMPT.format(
            current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            entity_types=[t.value for t in entity_types],
        )

        component_args = {
            "_component_name": "fargs.entities.extractor",
            "system_prompt": system_prompt,
            "output_model": build_output_model(
                build_entity_model(entity_types or DefaultEntityTypes)
            ),
        }
        if overwrite_config:
            component_args["agent_config"] = overwrite_config

        super().__init__(**component_args, **kwargs)

    async def aextract(self, nodes):
        entities = []

        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]
        async for task in tqdm_iterable(tasks, "Extracting entities..."):
            try:
                raw_results = await task
                entities.append({"entities": raw_results})
            except Exception as e:
                logger.exception(f"Error extracting entities: {e}")
                entities.append({"entities": None})

        return entities

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    @retry(
        (FargsExtractionError),  # TODO: use pydantic ai errors instead
        is_async=True,
        **RetryConfig.default(),
    )
    async def invoke_and_parse_results(self, node):
        entities = []

        extract_output = await self.invoke_llm(user_prompt=node.text)

        if extract_output.no_entities:
            return []

        for e in extract_output.entities:
            e._origin = node.node_id
            entities.append(e)

        return entities
