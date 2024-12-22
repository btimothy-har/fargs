import asyncio
import os
from collections import defaultdict
from typing import Literal

import tiktoken
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import ChunkNode
from llama_index.core.graph_stores import EntityNode
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.graph_stores import Relation
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from pydantic import Field
from pydantic import PrivateAttr
from pydantic_ai import Agent

from fargs.config import EMBEDDING_CONTEXT_LENGTH
from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import SUMMARY_CONTEXT_WINDOW
from fargs.config import LLMConfiguration
from fargs.models import DummyClaim
from fargs.models import DummyEntity
from fargs.models import Relationship
from fargs.prompts import SUMMARIZE_NODE_PROMPT
from fargs.utils import async_batch
from fargs.utils import sequential_task
from fargs.utils import token_limited_task
from fargs.utils import tqdm_iterable

from .base import default_llm_configuration

SUMMARIZE_NODE_MESSAGE = """
TYPE: {type}
TITLE: {title}
DESCRIPTION: {description}
"""


class GraphLoader(TransformComponent):
    config: LLMConfiguration = Field(default=default_llm_configuration)

    _graph_store: PropertyGraphStore | None = PrivateAttr(default=None)
    _embeddings: BaseEmbedding | None = PrivateAttr(default=None)
    _tokenizer: tiktoken.Encoding | None = PrivateAttr(default=None)
    _excluded_embed_metadata_keys: list[str] | None = PrivateAttr(default=None)
    _excluded_llm_metadata_keys: list[str] | None = PrivateAttr(default=None)

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        embeddings: BaseEmbedding,
        config: LLMConfiguration | None = None,
        excluded_embed_metadata_keys: list[str] | None = None,
        excluded_llm_metadata_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._graph_store = graph_store
        self._embeddings = embeddings
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

        if config:
            self.config = config

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        asyncio.run(self.acall(nodes, **kwargs))

    @token_limited_task(
        "cl100k_base",
        max_tokens_per_minute=os.getenv("FARGS_LLM_TOKEN_LIMIT", 100_000),
        max_requests_per_minute=os.getenv("FARGS_LLM_RATE_LIMIT", 1_000),
    )
    async def _embed_text(self, text: str):
        return await self._embeddings.aget_text_embedding(text)

    @token_limited_task(
        "cl100k_base",
        max_tokens_per_minute=os.getenv("FARGS_LLM_TOKEN_LIMIT", 100_000),
        max_requests_per_minute=os.getenv("FARGS_LLM_RATE_LIMIT", 1_000),
    )
    async def _summarize_node(
        self, node_type: Literal["entity", "relation"], title: str, description: str
    ):
        agent = Agent(
            model=self.config["model"],
            system_prompt=SUMMARIZE_NODE_PROMPT,
            name="fargs.node.summarizer",
            model_settings={"temperature": self.config["temperature"]},
        )

        return await agent.run(
            SUMMARIZE_NODE_MESSAGE.format(
                node_type=node_type,
                title=title,
                description=description,
            )
        )

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        await self._transform_entities(nodes)
        await self._transform_relations(nodes)

        all_claims = [
            claim for node in nodes for claim in node.metadata.get("claims", []) or []
        ]
        entities = (
            await asyncio.to_thread(
                self._graph_store.get,
                ids=list({c.subject_key for c in all_claims}),
            )
            if all_claims
            else None
        )
        entities_dict = {e.id: e for e in entities} if entities else {}

        transformed = []
        chunk_nodes = []
        rel_nodes = []

        tasks = [
            asyncio.create_task(self._transform_node(node, entities_dict))
            for node in nodes
        ]

        async for task in tqdm_iterable(
            asyncio.as_completed(tasks),
            "Transforming nodes...",
            total=len(tasks),
        ):
            t_node, t_chunk_nodes, t_rel_nodes = await task
            transformed.append(t_node)

            if t_chunk_nodes:
                chunk_nodes.extend(t_chunk_nodes)
            if t_rel_nodes:
                rel_nodes.extend(t_rel_nodes)

            if len(chunk_nodes) > min(PROCESSING_BATCH_SIZE, 1000):
                await asyncio.to_thread(self._graph_store.upsert_nodes, chunk_nodes)
                await asyncio.to_thread(self._graph_store.upsert_relations, rel_nodes)
                chunk_nodes = []
                rel_nodes = []

        if chunk_nodes:
            await asyncio.to_thread(self._graph_store.upsert_nodes, chunk_nodes)

        if rel_nodes:
            await asyncio.to_thread(self._graph_store.upsert_relations, rel_nodes)

        return transformed

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    async def _transform_node(
        self, node: BaseNode, entities_dict: dict[str, EntityNode] | None = None
    ) -> tuple[BaseNode, list[ChunkNode], list[Relation]]:
        async def _transform_claim(
            parent_node: BaseNode,
            claim: DummyClaim,
            subject_entity: EntityNode | None,
            object_entity: EntityNode | None,
        ) -> tuple[ChunkNode, list[Relation]]:
            chunk_node = ChunkNode(
                text=str(claim),
                id_=claim.key,
                label="claim",
                properties={
                    **claim.model_dump(
                        include={
                            "title",
                            "subject",
                            "object",
                            "claim_type",
                            "status",
                            "period",
                        }
                    ),
                    "source": parent_node.node_id,
                    "references": "; ".join(claim.sources),
                },
            )

            rel_nodes = []

            if subject_entity:
                rel_subject = Relation(
                    label="subject_of",
                    source_id=chunk_node.id,
                    target_id=claim.subject_key,
                )
                rel_nodes.append(rel_subject)

            if object_entity:
                rel_object = Relation(
                    label="object_of",
                    source_id=chunk_node.id,
                    target_id=claim.object_key,
                )
                rel_nodes.append(rel_object)

            return chunk_node, rel_nodes

        node.metadata.pop("entities", None)
        node.metadata.pop("relationships", None)

        claims = node.metadata.pop("claims", None)
        chunk_nodes = []
        rel_nodes = []

        if claims:
            tasks = [
                _transform_claim(
                    parent_node=node,
                    claim=c,
                    subject_entity=entities_dict.get(c.subject_key),
                    object_entity=entities_dict.get(c.object_key)
                    if c.object_key
                    else None,
                )
                for c in claims
            ]

            transformed_claims = await asyncio.gather(*tasks)
            for chunk_node, rel_nodes in transformed_claims:
                if self._graph_store.supports_vector_queries:
                    chunk_node.properties["embedding"] = await self._embed_text(
                        f"{chunk_node.label}: {chunk_node.text}"
                    )

                chunk_nodes.append(chunk_node)
                rel_nodes.extend(rel_nodes)

        node.metadata["chunk_id"] = node.node_id

        node.excluded_embed_metadata_keys = self._excluded_embed_metadata_keys
        node.excluded_llm_metadata_keys = self._excluded_llm_metadata_keys
        return node, chunk_nodes, rel_nodes

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    async def _transform_entities(self, nodes: list[BaseNode]):
        async def _transform(
            key: str,
            entities: list[DummyEntity],
            existing_entity: EntityNode | None = None,
        ) -> EntityNode:
            new_entity_values = {
                "id": key,
                "name": entities[0].name,
                "entity_type": entities[0].entity_type.value,
                "description": " ".join(f"{entity.description}" for entity in entities),
                "attributes": {
                    attr.name: attr.value
                    for entity in entities
                    for attr in entity.attributes
                },
                "references_": list({entity._origin for entity in entities}),
            }

            entity_values = {
                "id": key,
                "name": entities[0].name,
                "entity_type": entities[0].entity_type.value,
                "description": (
                    (
                        f"{existing_entity.properties.get('description', '')} "
                        f"{new_entity_values['description']}"
                    )
                    if existing_entity
                    else new_entity_values["description"]
                ),
                "attributes": ({
                    **(
                        {
                            k: v
                            for k, v in existing_entity.properties.items()
                            if k not in ["sources", "description", "references_"]
                        }
                        if existing_entity
                        else {}
                    ),
                    **new_entity_values["attributes"],
                }),
                "references_": (
                    new_entity_values["references_"]
                    + existing_entity.properties.get("sources", [])
                    + existing_entity.properties.get("references_", [])
                    if existing_entity
                    else new_entity_values["references_"]
                ),
            }

            desc_encoded = await asyncio.to_thread(
                self._tokenizer.encode, entity_values["description"]
            )
            if len(desc_encoded) > EMBEDDING_CONTEXT_LENGTH:
                desc = (
                    await asyncio.to_thread(
                        self._tokenizer.decode, desc_encoded[:SUMMARY_CONTEXT_WINDOW]
                    )
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else entity_values["description"]
                )
                try:
                    summary = await self._summarize_node(
                        node_type="entity",
                        title=entity_values["name"],
                        description=desc,
                    )
                except Exception:
                    entity_values["description"] = await asyncio.to_thread(
                        self._tokenizer.decode,
                        desc_encoded[:EMBEDDING_CONTEXT_LENGTH],
                    )
                else:
                    entity_values["description"] = summary.text

            entity_node = EntityNode(
                name=entity_values["name"],
                label=entity_values["entity_type"],
                properties={
                    "description": entity_values["description"],
                    "references_": list(set(entity_values["references_"])),
                    **entity_values["attributes"],
                },
            )

            return entity_node

        async def _embed(node: EntityNode) -> EntityNode:
            node_text = (
                f"{node.label}: {node.name} {node.properties.get('description', '')}"
            )
            node.properties["embedding"] = await self._embed_text(node_text)
            return node

        all_entities = defaultdict(list)
        entities: list[DummyEntity] = [
            entity
            for node in nodes
            for entity in node.metadata.pop("entities", []) or []
        ]
        for entity in entities:
            all_entities[entity.key].append(entity)

        existing_entities: list[EntityNode] = await asyncio.to_thread(
            self._graph_store.get, ids=list(all_entities.keys())
        )
        if existing_entities:
            existing_entities_dict = {e.id: e for e in existing_entities}
        else:
            existing_entities_dict = {}

        tasks = [
            asyncio.create_task(
                _transform(key, entities, existing_entities_dict.get(key))
            )
            for key, entities in list(all_entities.items())
        ]

        entities = []
        async for task in tqdm_iterable(
            asyncio.as_completed(tasks), "Transforming entities...", total=len(tasks)
        ):
            e = await task
            entities.append(e)

        final_entities = []
        if self._graph_store.supports_vector_queries:
            tasks = [asyncio.create_task(_embed(e)) for e in entities]

            async for task in tqdm_iterable(
                asyncio.as_completed(tasks),
                "Embedding entities...",
                total=len(tasks),
            ):
                final_entities.append(await task)
        else:
            final_entities = entities

        async for batch in async_batch(
            final_entities, min(PROCESSING_BATCH_SIZE, 1000)
        ):
            await asyncio.to_thread(self._graph_store.upsert_nodes, batch)

    @sequential_task(concurrent_tasks=PROCESSING_BATCH_SIZE)
    async def _transform_relations(self, nodes: list[BaseNode]):
        async def _transform(
            key: str,
            relations: list[Relationship],
            existing_relation: Relation | None = None,
        ) -> Relation:
            new_relation_values = {
                "id": key,
                "source_entity": relations[0].source_entity,
                "target_entity": relations[0].target_entity,
                "relation_type": relations[0].relation_type,
                "description": " ".join(
                    f"{relation.description}" for relation in relations
                ),
                "strength": sum(relation.strength for relation in relations)
                / len(relations),
                "references_": list(set([relation._origin for relation in relations])),
            }

            relation_values = {
                "id": key,
                "source_entity": new_relation_values["source_entity"],
                "target_entity": new_relation_values["target_entity"],
                "relation_type": new_relation_values["relation_type"],
                "description": (
                    (
                        f"{existing_relation.properties.get('description', '')} "
                        f"{new_relation_values['description']}"
                    )
                    if existing_relation
                    else new_relation_values["description"]
                ),
                "strength": (
                    (
                        existing_relation.properties.get("strength", 0)
                        + new_relation_values["strength"]
                    )
                    / 2
                    if existing_relation
                    else new_relation_values["strength"]
                ),
                "references_": (
                    new_relation_values["references_"]
                    + existing_relation.properties.get("references_", [])
                    if existing_relation
                    else new_relation_values["references_"]
                ),
            }

            desc_encoded = await asyncio.to_thread(
                self._tokenizer.encode, relation_values["description"]
            )
            if len(desc_encoded) > EMBEDDING_CONTEXT_LENGTH:
                desc = (
                    self._tokenizer.decode(desc_encoded[:SUMMARY_CONTEXT_WINDOW])
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else relation_values["description"]
                )
                try:
                    summary = await self._summarize_node(
                        node_type="relation",
                        title=(
                            f"{relation_values['source_entity']} -> "
                            f"{relation_values['relation_type']} -> "
                            f"{relation_values['target_entity']}"
                        ),
                        description=desc,
                    )
                except Exception:
                    relation_values["description"] = await asyncio.to_thread(
                        self._tokenizer.decode,
                        desc_encoded[:EMBEDDING_CONTEXT_LENGTH],
                    )
                else:
                    relation_values["description"] = summary.text

            relation_node = Relation(
                label=relation_values["relation_type"],
                source_id=relation_values["source_entity"],
                target_id=relation_values["target_entity"],
                properties={
                    "description": relation_values["description"],
                    "strength": relation_values["strength"],
                    "references_": list(set(relation_values["references_"])),
                },
            )

            return relation_node

        all_relationships = defaultdict(list)
        relationships = [
            relationship
            for node in nodes
            for relationship in node.metadata.pop("relationships", []) or []
        ]
        for relationship in relationships:
            all_relationships[relationship.key].append(relationship)

        existing_relationships: list[Relation] = await asyncio.to_thread(
            self._graph_store.get_triplets, ids=list(all_relationships.keys())
        )
        if existing_relationships:
            existing_relationships_dict = {
                r[1].id: r[1] for r in existing_relationships
            }
        else:
            existing_relationships_dict = {}

        tasks = [
            asyncio.create_task(
                _transform(key, relationships, existing_relationships_dict.get(key))
            )
            for key, relationships in list(all_relationships.items())
        ]

        relations = []
        async for task in tqdm_iterable(
            asyncio.as_completed(tasks),
            "Transforming relations...",
            total=len(tasks),
        ):
            r = await task
            relations.append(r)

            if len(relations) > min(PROCESSING_BATCH_SIZE, 1000):
                await asyncio.to_thread(self._graph_store.upsert_relations, relations)
                relations = []

        if relations:
            await asyncio.to_thread(self._graph_store.upsert_relations, relations)
