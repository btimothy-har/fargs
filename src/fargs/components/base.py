import os
from abc import ABC

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic_ai import Agent

from fargs.config import LLMConfiguration
from fargs.exceptions import FargsNoResponseError
from fargs.utils import token_limited_task

default_llm_configuration = LLMConfiguration(model="gpt-4o-mini", temperature=0.0)


class LLMPipelineComponent(BaseModel, ABC):
    agent_config: LLMConfiguration = Field(default=default_llm_configuration)
    system_prompt: str = Field(default="You are a helpful Assistant.")
    output_model: BaseModel | None = Field(default=None)
    _component_name: str = PrivateAttr(default="fargs.component")

    @property
    def agent(self) -> Agent:
        agent_params = {
            "model": self.agent_config["model"],
            "system_prompt": self.system_prompt,
            "name": self._component_name,
            "model_settings": {"temperature": self.agent_config["temperature"]},
        }
        if self.output_model:
            agent_params["result_type"] = self.output_model

        return Agent(**agent_params)

    @token_limited_task(
        "o200k_base",
        max_tokens_per_minute=os.getenv("FARGS_LLM_TOKEN_LIMIT", 100_000),
        max_requests_per_minute=os.getenv("FARGS_LLM_RATE_LIMIT", 1_000),
    )
    async def invoke_llm(self, **kwargs):
        llm_result = await self.agent.run(**kwargs)

        if len(llm_result.usage.response_tokens) == 0:
            raise FargsNoResponseError("LLM returned an empty response")

        return llm_result.data
