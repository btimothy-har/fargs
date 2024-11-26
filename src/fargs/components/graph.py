import asyncio
import os
from collections import defaultdict
from typing import Any
from typing import Literal

import ell
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
from retry_async import retry

from fargs.config import SUMMARY_CONTEXT_WINDOW
from fargs.config import default_extraction_llm
from fargs.config import default_retry_config
from fargs.exceptions import FargsLLMError
from fargs.models import DummyClaim
from fargs.models import DummyEntity
from fargs.models import Relationship
from fargs.prompts import SUMMARIZE_NODE_PROMPT
from fargs.utils import token_limited_task
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
    _tokenizer: tiktoken.Encoding | None = PrivateAttr(default=None)
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
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

        self.config = overwrite_config or default_extraction_llm

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        asyncio.run(self.acall(nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        await self._transform_entities(nodes)
        await self._transform_relations(nodes)

        transformed = []
        async for n in tqdm_iterable(nodes, "Transforming nodes..."):
            transformed_node = await self._transform_node(n)
            transformed.append(transformed_node)

        return transformed

    @retry(
        (FargsLLMError),
        is_async=True,
        **default_retry_config,
    )
    @token_limited_task(max_tokens=os.getenv("FARGS_LLM_TOKEN_LIMIT", 100_000))
    async def _embed_node(self, node: Any):
        try:
            return await self._embeddings.aget_text_embedding(str(node))
        except Exception as e:
            raise FargsLLMError(f"Failed to embed node: {e}") from e

    async def _transform_node(self, node: BaseNode, **kwargs) -> BaseNode:
        async def _transform_claim(
            parent_node: BaseNode, claim: DummyClaim
        ) -> tuple[ChunkNode, list[Relation]]:
            subject_entities = await self._graph_store.aget(ids=[claim.subject_key])

            object_entities = (
                await self._graph_store.aget(ids=[claim.object_key])
                if claim.object_key
                else []
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
                    "source": parent_node.node_id,
                    "references": "; ".join(claim.sources),
                },
            )

            if self._graph_store.supports_vector_queries:
                chunk_node.properties["embedding"] = await self._embed_node(
                    str(chunk_node)
                )

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

        node.metadata.pop("entities", None)
        node.metadata.pop("relationships", None)

        claims = node.metadata.pop("claims", None)

        if claims:
            chunk_nodes = []
            rel_nodes = []
            async for c in tqdm_iterable(
                claims, "Transforming claims...", disable=True
            ):
                chunk_node, rel_nodes = await _transform_claim(node, c)
                chunk_nodes.append(chunk_node)
                rel_nodes.extend(rel_nodes)

            await self._graph_store.aupsert_nodes(chunk_nodes)
            await self._graph_store.aupsert_relations(rel_nodes)

        node.metadata["chunk_id"] = node.node_id

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
                "references_": list(set([entity._origin for entity in entities])),
            }

            get_existing_entity = await self._graph_store.aget(ids=[key])

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

            desc_encoded = self._tokenizer.encode(entity_values["description"])
            if len(desc_encoded) > 6000:
                desc = (
                    self._tokenizer.decode(desc_encoded[:SUMMARY_CONTEXT_WINDOW])
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else entity_values["description"]
                )
                summary = await self.invoke_llm(
                    node_type="entity",
                    title=entity_values["name"],
                    description=desc,
                )
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

            if self._graph_store.supports_vector_queries:
                entity_node.properties["embedding"] = await self._embed_node(
                    str(entity_node)
                )

            return entity_node

        all_entities = defaultdict(list)
        entities = [
            entity
            for node in nodes
            for entity in node.metadata.pop("entities", []) or []
        ]
        for entity in entities:
            all_entities[entity.key].append(entity)

        entity_nodes = []
        async for key, entities in tqdm_iterable(
            all_entities.items(), "Transforming entities..."
        ):
            transformed = await _transform(key, entities)
            entity_nodes.append(transformed)

        await self._graph_store.aupsert_nodes(entity_nodes)

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
                "references_": list(set([relation._origin for relation in relations])),
            }

            get_existing_relation = await self._graph_store.aget_triplets(
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
                "references_": (
                    new_relation_values["references_"]
                    + existing_relation.properties.get("references_", [])
                    if existing_relation
                    else new_relation_values["references_"]
                ),
            }

            desc_encoded = self._tokenizer.encode(relation_values["description"])
            if len(desc_encoded) > 6000:
                desc = (
                    self._tokenizer.decode(desc_encoded[:SUMMARY_CONTEXT_WINDOW])
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else relation_values["description"]
                )
                summary = await self.invoke_llm(
                    node_type="relation",
                    title=(
                        f"{relation_values['source_entity']} -> "
                        f"{relation_values['relation_type']} -> "
                        f"{relation_values['target_entity']}"
                    ),
                    description=desc,
                )

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

        relation_nodes = []
        async for key, relationships in tqdm_iterable(
            all_relationships.items(), "Transforming relations..."
        ):
            transformed = await _transform(key, relationships)
            relation_nodes.append(transformed)

        await self._graph_store.aupsert_relations(relation_nodes)

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
