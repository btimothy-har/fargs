import asyncio
from collections import defaultdict
from typing import Literal

import ell
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import ChunkNode
from llama_index.core.graph_stores import EntityNode
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.graph_stores import Relation
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from pydantic import Field
from pydantic import PrivateAttr
from retry_async import retry

from fargs.config import default_extraction_llm
from fargs.config import default_retry_config
from fargs.exceptions import FargsLLMError
from fargs.models import DummyClaim
from fargs.models import DummyEntity
from fargs.models import Relationship
from fargs.prompts import SUMMARIZE_NODE_PROMPT
from fargs.utils import tqdm_iterable

from .base import LLMPipelineComponent

SUMMARIZE_NODE_MESSAGE = """
TYPE: {type}
TITLE: {title}
DESCRIPTION: {description}
"""


class GraphLoader(TransformComponent, LLMPipelineComponent):
    config: dict = Field(default_factory=dict)

    _graph_store: PropertyGraphStore | None = PrivateAttr(default=None)
    _embeddings: BaseEmbedding | None = PrivateAttr(default=None)
    _excluded_embed_metadata_keys: list[str] | None = PrivateAttr(default=None)
    _excluded_llm_metadata_keys: list[str] | None = PrivateAttr(default=None)

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        embeddings: BaseEmbedding,
        overwrite_config: dict | None = None,
        excluded_embed_metadata_keys: list[str] | None = None,
        excluded_llm_metadata_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._graph_store = graph_store
        self._embeddings = embeddings
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

        self.config = overwrite_config or default_extraction_llm

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        asyncio.run(self.acall(nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        await self._transform_entities(nodes)
        await self._transform_relations(nodes)

        tasks = [asyncio.create_task(self._transform_node(node)) for node in nodes]

        transformed = []
        async for task in tqdm_iterable(tasks, "Transforming nodes..."):
            transformed.append(await task)

        return transformed

    async def _transform_node(self, node: BaseNode, **kwargs) -> BaseNode:
        node.metadata.pop("entities", None)
        node.metadata.pop("relationships", None)

        claims = node.metadata.pop("claims", None)

        if claims:
            tasks = [
                asyncio.create_task(self._transform_claim(node, c)) for c in claims
            ]

            chunk_nodes = []
            rel_nodes = []
            for task in tasks:
                chunk_node, rel_nodes = await task
                chunk_nodes.append(chunk_node)
                rel_nodes.extend(rel_nodes)

            self._graph_store.upsert_nodes(chunk_nodes)
            self._graph_store.upsert_relations(rel_nodes)

        node.excluded_embed_metadata_keys = self._excluded_embed_metadata_keys
        node.excluded_llm_metadata_keys = self._excluded_llm_metadata_keys
        return node

    async def _transform_entities(self, nodes: list[BaseNode]):
        @retry(
            (FargsLLMError),
            is_async=True,
            **default_retry_config,
        )
        async def _transform(key: str, entities: list[DummyEntity]) -> EntityNode:
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
            }

            get_existing_entity = self._graph_store.get(ids=[key])

            existing_entity = get_existing_entity[0] if get_existing_entity else None

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
                "attributes": (
                    {
                        **(
                            {
                                k: v
                                for k, v in existing_entity.properties.items()
                                if k not in ["sources", "description"]
                            }
                            if existing_entity
                            else {}
                        ),
                        **new_entity_values["attributes"],
                    }
                ),
            }

            try:
                summary = await self.invoke_llm(
                    node_type="entity",
                    title=entity_values["name"],
                    description=entity_values["description"],
                )
            except FargsLLMError as e:
                raise FargsLLMError(f"Failed to invoke LLM: {e}") from e

            entity_values["description"] = summary.text

            entity_node = EntityNode(
                name=entity_values["name"],
                label=entity_values["entity_type"],
                properties={
                    "description": entity_values["description"],
                    **entity_values["attributes"],
                },
            )

            if self._graph_store.supports_vector_queries:
                entity_node.properties[
                    "embedding"
                ] = await self._embeddings.aget_text_embedding(str(entity_node))

            return entity_node

        all_entities = defaultdict(list)
        entities = [
            entity
            for node in nodes
            for entity in node.metadata.pop("entities", []) or []
        ]
        for entity in entities:
            all_entities[entity.key].append(entity)

        tasks = [
            asyncio.create_task(_transform(key, entities))
            for key, entities in all_entities.items()
        ]
        entity_nodes = []
        async for task in tqdm_iterable(tasks, "Transforming entities..."):
            entity_nodes.append(await task)

        self._graph_store.upsert_nodes(entity_nodes)

    async def _transform_relations(self, nodes: list[BaseNode]):
        @retry(
            (FargsLLMError),
            is_async=True,
            **default_retry_config,
        )
        async def _transform(key: str, relations: list[Relationship]) -> Relation:
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
            }

            get_existing_relation = self._graph_store.get_triplets(
                ids=[key],
            )

            existing_relation = (
                get_existing_relation[0] if get_existing_relation else None
            )

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
            }

            try:
                summary = await self.invoke_llm(
                    node_type="relation",
                    title=(
                        f"{relation_values['source_entity']} -> "
                        f"{relation_values['relation_type']} -> "
                        f"{relation_values['target_entity']}"
                    ),
                    description=relation_values["description"],
                )
            except FargsLLMError as e:
                raise FargsLLMError(f"Failed to invoke LLM: {e}") from e

            relation_values["description"] = summary.text

            relation_node = Relation(
                label=relation_values["relation_type"],
                source_id=relation_values["source_entity"],
                target_id=relation_values["target_entity"],
                properties={
                    "description": relation_values["description"],
                    "strength": relation_values["strength"],
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

        tasks = [
            asyncio.create_task(_transform(key, relationships))
            for key, relationships in all_relationships.items()
        ]
        relation_nodes = []
        async for task in tqdm_iterable(tasks, "Transforming relations..."):
            relation_nodes.append(await task)

        self._graph_store.upsert_relations(relation_nodes)

    async def _transform_claim(
        self, parent_node: BaseNode, claim: DummyClaim
    ) -> tuple[ChunkNode, list[Relation]]:
        subject_entities = self._graph_store.get(ids=[claim.subject_key])

        object_entities = (
            self._graph_store.get(ids=[claim.object_key]) if claim.object_key else []
        )

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
                "source": parent_node.metadata.get("doc_id", parent_node.node_id),
                "references": "; ".join(claim.sources),
            },
        )

        if self._graph_store.supports_vector_queries:
            chunk_node.properties[
                "embedding"
            ] = await self._embeddings.aget_text_embedding(str(chunk_node))

        rel_nodes = []

        if subject_entities:
            rel_subject = Relation(
                label="subject_of",
                source_id=chunk_node.id,
                target_id=subject_entities[0].id,
            )
            rel_nodes.append(rel_subject)

        if object_entities:
            rel_object = Relation(
                label="object_of",
                source_id=chunk_node.id,
                target_id=object_entities[0].id,
            )
            rel_nodes.append(rel_object)

        return chunk_node, rel_nodes

    def _construct_function(self):
        @ell.complex(**self.config)
        def summarize_node(
            node_type: Literal["entity", "relation"],
            title: str,
            description: str,
        ):
            return [
                ell.system(SUMMARIZE_NODE_PROMPT),
                ell.user(
                    SUMMARIZE_NODE_MESSAGE.format(
                        type=node_type, title=title, description=description
                    )
                ),
            ]

        return summarize_node
