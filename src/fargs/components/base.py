import asyncio
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

from pydantic import BaseModel
from pydantic import PrivateAttr

from fargs.exceptions import FargsNoResponseError
from fargs.utils import rate_limited_task


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

    @rate_limited_task(max_rate=os.getenv("FARGS_LLM_RATE_LIMIT", 10))
    async def invoke_llm(self, **kwargs):
        llm_result = await asyncio.to_thread(self.llm_fn, **kwargs)
        if len(llm_result.text) == 0:
            raise FargsNoResponseError("LLM returned an empty response")
        return llm_result
