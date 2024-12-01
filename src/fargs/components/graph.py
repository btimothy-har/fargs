import asyncio
from collections import defaultdict
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

from fargs.config import EMBEDDING_CONTEXT_LENGTH
from fargs.config import PROCESSING_BATCH_SIZE
from fargs.config import SUMMARY_CONTEXT_WINDOW
from fargs.config import default_extraction_llm
from fargs.models import DummyClaim
from fargs.models import DummyEntity
from fargs.models import Relationship
from fargs.prompts import SUMMARIZE_NODE_PROMPT
from fargs.utils import async_batch
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

        async for node in tqdm_iterable(
            nodes,
            "Transforming nodes...",
        ):
            t = await asyncio.to_thread(self._transform_node, node)
            transformed.append(t)

        return transformed

    def _transform_node(self, node: BaseNode, **kwargs) -> BaseNode:
        def _transform_claim(
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

            if self._graph_store.supports_vector_queries:
                chunk_node.properties["embedding"] = (
                    self._embeddings.get_text_embedding(str(claim))
                )

            rel_nodes = []

            if subject_entity:
                rel_subject = Relation(
                    label="subject_of",
                    source_id=chunk_node.id,
                    target_id=subject_entity.id,
                )
                rel_nodes.append(rel_subject)

            if object_entity:
                rel_object = Relation(
                    label="object_of",
                    source_id=chunk_node.id,
                    target_id=object_entity.id,
                )
                rel_nodes.append(rel_object)

            return chunk_node, rel_nodes

        node.metadata.pop("entities", None)
        node.metadata.pop("relationships", None)

        claims = node.metadata.pop("claims", None)

        if claims:
            subject_entities: list[EntityNode] = self._graph_store.get(
                ids=[c.subject_key for c in claims]
            )
            subject_entities_dict = {e.id: e for e in subject_entities}

            object_entities: list[EntityNode] = self._graph_store.get(
                ids=[c.object_key for c in claims if c.object_key],
            )
            object_entities_dict = {e.id: e for e in object_entities}

            chunk_nodes = []
            rel_nodes = []

            for c in claims:
                chunk_node, rel_nodes = _transform_claim(
                    parent_node=node,
                    claim=c,
                    subject_entity=subject_entities_dict.get(c.subject_key),
                    object_entity=object_entities_dict.get(c.object_key),
                )
                chunk_nodes.append(chunk_node)
                rel_nodes.extend(rel_nodes)

            self._graph_store.upsert_nodes(chunk_nodes)
            self._graph_store.upsert_relations(rel_nodes)

        node.metadata["chunk_id"] = node.node_id

        node.excluded_embed_metadata_keys = self._excluded_embed_metadata_keys
        node.excluded_llm_metadata_keys = self._excluded_llm_metadata_keys
        return node

    async def _transform_entities(self, nodes: list[BaseNode]):
        def _transform(
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
                "references_": list(set([entity._origin for entity in entities])),
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

            desc_encoded = self._tokenizer.encode(entity_values["description"])
            if len(desc_encoded) > EMBEDDING_CONTEXT_LENGTH:
                desc = (
                    self._tokenizer.decode(desc_encoded[:SUMMARY_CONTEXT_WINDOW])
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else entity_values["description"]
                )
                try:
                    summary = self.llm_fn(
                        node_type="entity",
                        title=entity_values["name"],
                        description=desc,
                    )
                except Exception:
                    entity_values["description"] = self._tokenizer.decode(
                        desc_encoded[:EMBEDDING_CONTEXT_LENGTH]
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

            if self._graph_store.supports_vector_queries:
                node_text = (
                    f"{entity_values['entity_type']}: {entity_values['name']} "
                    f"{entity_values['description']}"
                )
                entity_node.properties["embedding"] = (
                    self._embeddings.get_text_embedding(node_text)
                )

            return entity_node

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

        entities = []
        async for key, entities in tqdm_iterable(
            list(all_entities.items()), "Transforming entities..."
        ):
            e = await asyncio.to_thread(
                _transform, key, entities, existing_entities_dict.get(key)
            )
            entities.append(e)

        async for batch in async_batch(
            entities, batch_size=min(PROCESSING_BATCH_SIZE, 1000)
        ):
            await asyncio.to_thread(self._graph_store.upsert_nodes, batch)

    async def _transform_relations(self, nodes: list[BaseNode]):
        def _transform(
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

            desc_encoded = self._tokenizer.encode(relation_values["description"])
            if len(desc_encoded) > EMBEDDING_CONTEXT_LENGTH:
                desc = (
                    self._tokenizer.decode(desc_encoded[:SUMMARY_CONTEXT_WINDOW])
                    if len(desc_encoded) > SUMMARY_CONTEXT_WINDOW
                    else relation_values["description"]
                )
                try:
                    summary = self.llm_fn(
                        node_type="relation",
                        title=(
                            f"{relation_values['source_entity']} -> "
                            f"{relation_values['relation_type']} -> "
                            f"{relation_values['target_entity']}"
                        ),
                        description=desc,
                    )
                except Exception:
                    relation_values["description"] = self._tokenizer.decode(
                        desc_encoded[:EMBEDDING_CONTEXT_LENGTH]
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
            existing_relationships_dict = {r.id: r for r in existing_relationships}
        else:
            existing_relationships_dict = {}

        relations = []

        async for key, relationships in tqdm_iterable(
            list(all_relationships.items()),
            "Transforming relations...",
        ):
            r = await asyncio.to_thread(
                _transform, key, relationships, existing_relationships_dict.get(key)
            )
            relations.append(r)

        async for batch in async_batch(
            relations, batch_size=min(PROCESSING_BATCH_SIZE, 1000)
        ):
            await asyncio.to_thread(self._graph_store.upsert_relations, batch)

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
