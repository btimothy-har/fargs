import asyncio
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

import pandas as pd
from aiolimiter import AsyncLimiter
from langchain_text_splitters import CharacterTextSplitter
from tiktoken import encoding_for_model

from fargs.llm import OpenAIChatModels
from fargs.llm import OpenAIEmbeddingModels
from fargs.llm import get_chat_client
from fargs.llm import get_embeddings
from fargs.models import DefaultEntityTypes
from fargs.models import Document
from fargs.pipeline import tasks

from .progress import ProgressReporter


class GraphPipeline:
    def __init__(
        self,
        chat_model: str = OpenAIChatModels.GPT_4O_MINI.value,
        embedding_model: str = OpenAIEmbeddingModels.TEXT_EMBEDDING_3_SMALL.value,
        token_limit: int = 1_000_000,
        entity_types: Enum = DefaultEntityTypes,
        df_documents: pd.DataFrame = None,
        df_text_units: pd.DataFrame = None,
        df_entities: pd.DataFrame = None,
        df_relationships: pd.DataFrame = None,
        df_claims: pd.DataFrame = None,
    ):
        self.llm = get_chat_client(chat_model)
        self.embedding = get_embeddings(embedding_model)
        self.encoder = encoding_for_model(chat_model)
        self.chunker = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoder.name,
            chunk_size=256,
            chunk_overlap=25,
        )
        self.entity_types = entity_types

        self.df_documents = df_documents if df_documents is not None else pd.DataFrame()
        self.df_text_units = (
            df_text_units if df_text_units is not None else pd.DataFrame()
        )
        self.df_entities = df_entities if df_entities is not None else pd.DataFrame()
        self.df_relationships = (
            df_relationships if df_relationships is not None else pd.DataFrame()
        )
        self.df_claims = df_claims if df_claims is not None else pd.DataFrame()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._df_lock = defaultdict(asyncio.Lock)
        self._limiter = AsyncLimiter(token_limit)
        self._semaphore = asyncio.Semaphore(4)

    def ingest_document(self, document_dict: dict):
        self._loop.run_until_complete(self._ingest_document(document_dict))

    def batch_documents(self, documents: list[dict]):
        self._loop.run_until_complete(self._batch_documents(documents))

    async def _batch_documents(self, documents: list[dict]):
        tasks = []

        for i, document in enumerate(documents):
            task = asyncio.create_task(self._ingest_document(document, i))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            await task

    async def _ingest_document(self, document_dict: dict, iter_num: int = 0):
        document = Document(**document_dict)

        progress = ProgressReporter(document.title)
        progress.start(iter_num)

        async with self._df_lock["documents"]:
            if not self.df_documents.empty:
                if document.content_hash in self.df_documents["content_hash"].values:
                    return

            self.df_documents = pd.concat(
                [self.df_documents, pd.DataFrame([document.model_dump()])],
                ignore_index=True,
            ).drop_duplicates(subset=["doc_id"], keep="last")

        text_units = await tasks.build_text_units(self, progress, document)
        entities = await tasks.extract_entities(self, progress, text_units)
        entities = await tasks.resolve_entities_by_name(self, progress, entities)
        entities = await tasks.resolve_document_entities(self, progress, entities)

        async with self._df_lock["entities"]:
            if self.df_entities.empty:
                self.df_entities = pd.concat(
                    [self.df_entities, entities], ignore_index=True
                )
            else:
                self.df_entities = await tasks.resolve_global_entities(
                    self, progress, entities
                )

        relationships = await tasks.extract_relationships(
            self, progress, text_units, entities
        )
        claims = await tasks.extract_claims(self, progress, text_units, entities)

        async with self._df_lock["text_units"]:
            if self.df_text_units.empty:
                self.df_text_units = text_units
            else:
                self.df_text_units = self.df_text_units[
                    self.df_text_units["doc_id"] != document.doc_id
                ]
                self.df_text_units = pd.concat(
                    [self.df_text_units, text_units], ignore_index=True
                )

        async with self._df_lock["relationships"]:
            self.df_relationships = pd.concat(
                [self.df_relationships, relationships], ignore_index=True
            )

        async with self._df_lock["claims"]:
            self.df_claims = pd.concat([self.df_claims, claims], ignore_index=True)

    def _encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def _split_text(self, text: str) -> list[str]:
        return self.chunker.split_text(text)

    async def _embed_text(self, text: str) -> list[float]:
        input_tokens = self._encode(text)

        async with self._semaphore:
            await self._limiter.acquire(len(input_tokens))
            embeddings = await self.embedding.aembed_query(text)

        return embeddings

    async def _invoke_llm(self, task: Callable, messages: list[Any]):
        input_tokens = len(self._encode(str(messages)))

        async with self._semaphore:
            await self._limiter.acquire(input_tokens)
            response = await task(messages)
        return response
