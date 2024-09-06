import asyncio

import pandas as pd

from fargs.data import BaseGraphData
from fargs.data import ClaimsParquetData
from fargs.data import DocumentsParquetData
from fargs.data import EntitiesParquetData
from fargs.data import RelationshipsParquetData
from fargs.data import TextUnitsParquetData
from fargs.models import Document
from fargs.models import TextUnit

from .base import FargsConfig
from .documents import DocumentTasks as FargsDocuments
from .graph import GraphTasks as FargsGraph


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class Fargs(FargsDocuments, FargsGraph):
    def __init__(self, config: FargsConfig):
        super().__init__(config)
        self.config = config

        config_dict = config.model_dump()
        self._documents = DocumentsParquetData(config=config_dict)
        self._text_units = TextUnitsParquetData(config=config_dict)
        self._entities = EntitiesParquetData(config=config_dict)
        self._relationships = RelationshipsParquetData(config=config_dict)
        self._claims = ClaimsParquetData(config=config_dict)
        self._graph = BaseGraphData(config=config_dict)

    @property
    def documents(self) -> DocumentsParquetData:
        return self._documents

    @property
    def text_units(self) -> TextUnitsParquetData:
        return self._text_units

    @property
    def entities(self) -> EntitiesParquetData:
        return self._entities

    @property
    def relationships(self) -> RelationshipsParquetData:
        return self._relationships

    @property
    def claims(self) -> ClaimsParquetData:
        return self._claims

    @property
    def graph(self) -> BaseGraphData:
        return self._graph

    def write_data(self):
        self._loop.run_until_complete(self.async_write_data())

    def add_document(self, document_dict: dict):
        self._loop.run_until_complete(self.async_add_document(document_dict))

    def batch_documents(self, documents: list[dict]):
        self._loop.run_until_complete(self.async_batch_documents(documents))

    async def async_write_data(self):
        await self.documents.write()
        await self.documents.write(fmt="csv")
        await self.text_units.write()
        await self.text_units.write(fmt="csv")
        await self.entities.write()
        await self.entities.write(fmt="csv")
        await self.relationships.write()
        await self.relationships.write(fmt="csv")
        await self.claims.write()
        await self.claims.write(fmt="csv")

    async def async_batch_documents(self, documents: list[dict]):
        tasks = []

        for document in documents:
            task = asyncio.create_task(self.async_add_document(document))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            await task

    async def async_add_document(self, document_dict: dict):
        document = Document(**document_dict)

        doc_continue = await self.documents.insert(document)
        if not doc_continue:
            return

        df_text_units = await self.build_text_units(document)

        get_entities = [
            asyncio.create_task(self.extract_entities(TextUnit(**unit.to_dict())))
            for _, unit in df_text_units.iterrows()
        ]
        all_df_entities = await asyncio.gather(*get_entities)
        df_entities = pd.concat(all_df_entities, ignore_index=True)

        df_entities = await self.resolve_entities_by_name(document, df_entities)
        df_entities = await self.resolve_document_entities(document, df_entities)

        df_relationships = await self.extract_relationships(df_text_units, df_entities)
        df_claims = await self.extract_claims(df_text_units, df_entities)

        insert = [
            self.text_units.insert(df_text_units),
            self.entities.insert(df_entities),
            self.relationships.insert(df_relationships),
            self.claims.insert(df_claims),
        ]
        await asyncio.gather(*insert)
