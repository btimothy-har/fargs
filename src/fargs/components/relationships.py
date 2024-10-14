import asyncio
import json

import ell
import pydantic
from llama_index.core.extractors import BaseExtractor
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from retry_async import retry

from fargs.config import default_extraction_llm
from fargs.exceptions import FargsExtractionError
from fargs.exceptions import FargsLLMError
from fargs.models import Relationship
from fargs.prompts import EXTRACT_RELATIONSHIPS_PROMPT
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent

RELATIONSHIP_EXTRACTION_MESSAGE = """
ENTITIES
----------
{entities_json}

TEXT
----------
{text_unit}
"""


class RelationshipOutput(BaseModel):
    relationships: list[Relationship] = Field(
        title="Relationships",
        description="List of relationships identified.",
    )


class RelationshipExtractor(BaseExtractor, LLMPipelineComponent):
    config: dict = Field(default_factory=dict)

    _prompt: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        prompt: str = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._prompt = prompt
        self.config = overwrite_config or default_extraction_llm

        if "gpt-4o" in self.config["model"]:
            self.config["response_format"] = RelationshipOutput
        else:
            self.config["response_format"] = {"type": "json_object"}

    @property
    def prompt(self):
        if self._prompt is None:
            self._prompt = EXTRACT_RELATIONSHIPS_PROMPT.format(
                output_schema=RelationshipOutput.model_json_schema()
            )
        return self._prompt

    def _construct_function(self):
        @ell.complex(**self.config)
        def extract_relationships(entities_json: str, text_unit: str):
            return [
                ell.system(self.prompt),
                ell.user(
                    RELATIONSHIP_EXTRACTION_MESSAGE.format(
                        entities_json=entities_json,
                        text_unit=text_unit,
                    )
                ),
            ]

        return extract_relationships

    async def aextract(self, nodes):
        relationships = []

        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]
        async for task in tqdm_iterable(tasks, "Extracting relationships"):
            try:
                raw_results = await task
                relationships.append({"relationships": raw_results})
            except Exception:
                relationships.append({"relationships": None})

        return relationships

    @retry(
        (FargsExtractionError, FargsLLMError),
        is_async=True,
        tries=3,
        delay=1,
        backoff=2,
    )
    async def invoke_and_parse_results(self, node):
        if not node.metadata.get("entities"):
            return None

        relationships = []
        entities_json = "\n".join(
            [
                m.model_dump_json(
                    include={
                        "name",
                        "entity_type",
                        "description",
                    }
                )
                for m in node.metadata["entities"]
            ]
        )

        try:
            raw_result = await self.invoke_llm(
                entities_json=entities_json,
                text_unit=node.text,
            )
        except Exception as e:
            raise FargsLLMError(f"Failed to invoke LLM: {e}") from e

        if "gpt-4o" in self.config["model"]:
            parsed_output = raw_result.parsed
            relationships = parsed_output.relationships
        else:
            try:
                raw_relationships = json.loads(raw_result.text_only)
            except json.JSONDecodeError as e:
                raise FargsExtractionError(
                    f"Failed to parse relationships from LLM output: {e}\n\n"
                    f"{raw_result.text_only}"
                ) from e
            else:
                for r in raw_relationships["relationships"]:
                    if isinstance(r, dict):
                        try:
                            relationships.append(Relationship.model_validate(r))
                        except pydantic.ValidationError as e:
                            raise FargsExtractionError(
                                f"Failed to validate relationship: {e}\n\n{r}"
                            ) from e
        return relationships
