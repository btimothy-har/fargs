import asyncio

from llama_index.core.extractors import BaseExtractor
from pydantic import BaseModel
from pydantic import Field

from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import LLMConfiguration
from fargs.models import Relationship
from fargs.prompts import EXTRACT_RELATIONSHIPS_PROMPT
from fargs.utils import logger
from fargs.utils import sequential_task
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent

RELATIONSHIP_EXTRACTION_MESSAGE = """
<entities>
{entities_text}
</entities>

<text>
{text_unit}
</text>
"""


class RelationshipOutput(BaseModel):
    relationships: list[Relationship] = Field(
        title="Relationships",
        description=(
            "List of relationships identified. If there are no relationships, you may"
            " respond with an empty list."
        ),
    )


class RelationshipExtractor(BaseExtractor, LLMPipelineComponent):
    def __init__(
        self,
        prompt: str = None,
        overwrite_config: LLMConfiguration | None = None,
        **kwargs,
    ):
        component_args = {
            "component_name": "fargs.relationships.extractor",
            "system_prompt": prompt or EXTRACT_RELATIONSHIPS_PROMPT,
            "output_model": RelationshipOutput,
        }

        if overwrite_config:
            component_args["agent_config"] = overwrite_config

        super().__init__(**component_args, **kwargs)

    async def aextract(self, nodes):
        relationships = []
        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]

        async for task in tqdm_iterable(
            tasks,
            "Extracting relationships...",
        ):
            try:
                raw_results = await task
                relationships.append({"relationships": raw_results})
            except Exception as e:
                logger.exception(f"Error extracting relationships: {e}")
                relationships.append({"relationships": None})

        return relationships

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    async def invoke_and_parse_results(self, node):
        if not node.metadata.get("entities"):
            return None

        relationships = []
        entities_text = "\n\n".join([
            f"NAME: {m.name}, TYPE: {m.entity_type}, DESCRIPTION: {m.description}"
            for m in node.metadata["entities"]
        ])

        extract_output = await self.invoke_llm(
            user_prompt=RELATIONSHIP_EXTRACTION_MESSAGE.format(
                entities_text=entities_text,
                text_unit=node.text,
            )
        )

        if not extract_output.relationships:
            return []

        for r in extract_output.relationships:
            r._origin = node.node_id
            relationships.append(r)

        return relationships
