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
from fargs.models import DefaultClaimTypes
from fargs.models import build_claim_model
from fargs.prompts import EXTRACT_CLAIMS_PROMPT
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent

CLAIM_EXTRACTION_MESSAGE = """
ENTITIES
----------
{entities_json}

TEXT
----------
{text_unit}
"""


def build_output_model(base_model: BaseModel):
    class ClaimOutput(BaseModel):
        claims: list[base_model] = Field(
            title="Claims", description="List of claims identified."
        )

    return ClaimOutput


class ClaimsExtractor(BaseExtractor, LLMPipelineComponent):
    config: dict = Field(default_factory=dict)

    _prompt: str | None = PrivateAttr(default=None)
    _claim_types: Enum | None = PrivateAttr(default=None)
    _claim_model: BaseModel = PrivateAttr(default_factory=lambda: BaseModel)
    _output_model: BaseModel = PrivateAttr(default_factory=lambda: BaseModel)

    def __init__(
        self,
        prompt: str = None,
        claim_types: Enum = None,
        overwrite_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._prompt = prompt
        self._claim_types = claim_types or DefaultClaimTypes

        self._claim_model = build_claim_model(self._claim_types)
        self._output_model = build_output_model(self._claim_model)

        self.config = overwrite_config or default_extraction_llm

        if "gpt-4o" in self.config["model"]:
            self.config["response_format"] = self._output_model
        else:
            self.config["response_format"] = {"type": "json_object"}

    @property
    def prompt(self):
        if self._prompt is None:
            return EXTRACT_CLAIMS_PROMPT.format(
                current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                claim_types=[t.value for t in self._claim_types],
                output_schema=self._output_model.model_json_schema(),
            )
        return self._prompt

    def _construct_function(self):
        @ell.complex(**self.config)
        def extract_claims(entities_json: str, text_unit: str):
            return [
                ell.system(self.prompt),
                ell.user(
                    CLAIM_EXTRACTION_MESSAGE.format(
                        entities_json=entities_json,
                        text_unit=text_unit,
                    )
                ),
            ]

        return extract_claims

    async def aextract(self, nodes):
        claims = []

        tasks = [
            asyncio.create_task(self.invoke_and_parse_results(node)) for node in nodes
        ]
        async for task in tqdm_iterable(tasks, "Extracting claims"):
            try:
                raw_results = await task
                claims.append({"claims": raw_results})
            except Exception:
                claims.append({"claims": None})

        return claims

    @retry(
        (FargsLLMError, FargsExtractionError),
        is_async=True,
        **default_retry_config,
    )
    async def invoke_and_parse_results(self, node):
        if not node.metadata.get("entities"):
            return None

        claims = []
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
            claims = parsed_output.claims
        else:
            try:
                raw_claims = json.loads(raw_result.text_only)
            except json.JSONDecodeError as e:
                raise FargsExtractionError(
                    f"Failed to parse claims JSON from LLM output: {e}\n\n"
                    f"{raw_result.text_only}"
                ) from e
            else:
                try:
                    for r in raw_claims["claims"]:
                        if isinstance(r, dict):
                            try:
                                claims.append(self._claim_model.model_validate(r))
                            except pydantic.ValidationError as e:
                                raise FargsExtractionError(
                                    f"Failed to validate entity: {e}\n\n{r}"
                                ) from e
                except KeyError as e:
                    raise FargsExtractionError(
                        f"Failed to parse claims from LLM output: {raw_claims}"
                    ) from e
        return claims
