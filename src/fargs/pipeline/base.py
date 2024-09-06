import asyncio
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from functools import cached_property
from typing import Any

from aiolimiter import AsyncLimiter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel
from pydantic import field_validator
from tiktoken import Encoding
from tiktoken import encoding_for_model

from fargs.data import EntitiesParquetData
from fargs.data import RelationshipsParquetData
from fargs.llm import OpenAIChatModels
from fargs.llm import OpenAIEmbeddingModels
from fargs.llm import get_chat_client
from fargs.llm import get_chunk_strategy
from fargs.llm import get_embedding_client
from fargs.models import DefaultEntityTypes


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class FargsConfig(BaseModel):
    working_dir: str
    chat_model: str = OpenAIChatModels.GPT_4O_MINI.value
    embedding_model: str = OpenAIEmbeddingModels.TEXT_EMBEDDING_3_SMALL.value
    token_limit: int = 1_000_000
    task_concurrency: int = 4
    entity_types: Enum = DefaultEntityTypes
    chunk_strategy: str = "tokens"
    chunk_size: int = 256
    chunk_overlap: int = 25

    @field_validator("chat_model", mode="before")
    @classmethod
    def validate_chat_model(cls, value):
        if value not in [t.value for t in OpenAIChatModels]:
            raise ValueError(f"Invalid chat model: {value}")
        return value

    @field_validator("embedding_model", mode="before")
    @classmethod
    def validate_embedding_model(cls, value):
        if value not in [t.value for t in OpenAIEmbeddingModels]:
            raise ValueError(f"Invalid embedding model: {value}")
        return value


class BaseFargs(ABC):
    def __init__(self, config: FargsConfig):
        self.config = config
        self._loop = get_event_loop()
        self._limiter = AsyncLimiter(config.token_limit)
        self._semaphore = asyncio.Semaphore(config.task_concurrency)

    @property
    @abstractmethod
    def entities(self) -> EntitiesParquetData:
        pass

    @property
    @abstractmethod
    def relationships(self) -> RelationshipsParquetData:
        pass

    @cached_property
    def encoder(self) -> Encoding:
        return encoding_for_model(self.config.chat_model)

    @cached_property
    def embedding(self) -> OpenAIEmbeddings:
        return get_embedding_client(self.config.embedding_model)

    @cached_property
    def llm(self) -> ChatOpenAI:
        return get_chat_client(self.config.chat_model)

    @cached_property
    def chunker(self) -> CharacterTextSplitter:
        return get_chunk_strategy(
            model=self.config.chat_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    async def embed_text(self, text: str) -> list[float]:
        input_tokens = self.encode(text)

        async with self._semaphore:
            await self._limiter.acquire(len(input_tokens))
            embeddings = await self.embedding.aembed_query(text)

        return embeddings

    async def invoke_llm(self, task: Callable, messages: list[Any]):
        input_tokens = len(self.encode(str(messages)))

        async with self._semaphore:
            await self._limiter.acquire(input_tokens)
            response = await task(messages)
        return response
