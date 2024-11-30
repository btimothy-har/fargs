import asyncio
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

from pydantic import BaseModel
from pydantic import PrivateAttr

from fargs.exceptions import FargsLLMError
from fargs.exceptions import FargsNoResponseError
from fargs.utils import token_limited_task


class LLMPipelineComponent(BaseModel, ABC):
    _llm_fn: Callable | None = PrivateAttr(default=None)

    @property
    @abstractmethod
    def _construct_function(self):
        return

    @property
    def llm_fn(self) -> Callable:
        if not self._llm_fn:
            self._llm_fn = self._construct_function()
        return self._llm_fn

    @token_limited_task(
        "o200k_base", max_tokens=os.getenv("FARGS_LLM_TOKEN_LIMIT", 100_000)
    )
    async def invoke_llm(self, **kwargs):
        try:
            llm_result = await asyncio.to_thread(self.llm_fn, **kwargs)
        except Exception as e:
            raise FargsLLMError(f"Failed to invoke LLM: {e}") from e

        if len(llm_result.text) == 0:
            raise FargsNoResponseError("LLM returned an empty response")
        return llm_result
