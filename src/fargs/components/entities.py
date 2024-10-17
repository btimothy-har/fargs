import asyncio
import json
from datetime import UTC
from datetime import datetime
from enum import Enum

import ell
import pydantic
from llama_index.core.extractors import BaseExtractor
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from retry_async import retry

from fargs.config import default_extraction_llm
from fargs.config import default_retry_config
from fargs.exceptions import FargsExtractionError
from fargs.exceptions import FargsLLMError
from fargs.models import DefaultEntityTypes
from fargs.models import build_entity_model
from fargs.prompts import EXTRACT_ENTITIES_PROMPT
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent


def build_output_model(base_model: BaseModel):
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
    config: dict = Field(default_factory=dict)

    _prompt: str | None = PrivateAttr(default=None)
    _entity_types: Enum | None = PrivateAttr(default=None)
    _entity_model: BaseModel = PrivateAttr(default_factory=lambda: BaseModel)
    _output_model: BaseModel = PrivateAttr(default_factory=lambda: BaseModel)

    def __init__(
        self,
        prompt: str = None,
        entity_types: Enum = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._prompt = prompt
        self._entity_types = entity_types or DefaultEntityTypes
        self._entity_model = build_entity_model(self._entity_types)
        self._output_model = build_output_model(self._entity_model)

        self.config = overwrite_config or default_extraction_llm

        if "gpt-4o" in self.config["model"]:
            self.config["response_format"] = self._output_model
        else:
            self.config["response_format"] = {"type": "json_object"}

    @property
    def prompt(self):
        if self._prompt is None:
            return EXTRACT_ENTITIES_PROMPT.format(
                current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                entity_types=[t.value for t in self._entity_types],
                output_schema=self._output_model.model_json_schema(),
            )
        return self._prompt

    def _construct_function(self):
        @ell.complex(**self.config)
        def extract_entities(node_text: str):
            return [
                ell.system(self.prompt),
                ell.user(node_text),
            ]

        return extract_entities

    async def aextract(self, nodes):
        entities = []

        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]
        async for task in tqdm_iterable(tasks, "Extracting entities"):
            try:
                raw_results = await task
                entities.append({"entities": raw_results})
            except Exception as e:
                print(f"Error: {e}")
                entities.append({"entities": None})

        return entities

    @retry(
        (FargsExtractionError, FargsLLMError),
        is_async=True,
        **default_retry_config,
    )
    async def invoke_and_parse_results(self, node):
        entities = []

        raw_result = await self.invoke_llm(node_text=node.text)

        if "gpt-4o" in self.config["model"]:
            parsed_output = raw_result.parsed
            if parsed_output.no_entities:
                return []
            entities = parsed_output.entities
        else:
            try:
                raw_entities = json.loads(raw_result.text_only)
            except json.JSONDecodeError as e:
                raise FargsExtractionError(
                    f"Failed to parse entities from LLM output: {e}\n\n"
                    f"{raw_result.text_only}"
                ) from e
            else:
                try:
                    if raw_entities["no_entities"]:
                        return []
                    for r in raw_entities["entities"]:
                        if isinstance(r, dict):
                            try:
                                entities.append(self._entity_model.model_validate(r))
                            except pydantic.ValidationError as e:
                                raise FargsExtractionError(
                                    f"Failed to validate entity: {e}\n\n{r}"
                                ) from e
                except KeyError as e:
                    raise FargsExtractionError(
                        f"Failed to parse entities from LLM output: {raw_entities}"
                    ) from e
        return entities
