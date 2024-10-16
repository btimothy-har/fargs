import asyncio
import ell
from typing import Literal
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import ChunkNode
from llama_index.core.graph_stores import EntityNode
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.graph_stores import Relation
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from pydantic import Field
from pydantic import PrivateAttr

from fargs.config import default_extraction_llm
from fargs.models import DummyClaim
from fargs.models import DummyEntity
from fargs.models import Relationship
from fargs.utils import sequential_task
from fargs.utils import tqdm_iterable
from fargs.prompts import SUMMARIZE_NODE_PROMPT
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
        tasks = [asyncio.create_task(self.transform_node(node)) for node in nodes]

        transformed = []
        async for task in tqdm_iterable(tasks, "Transforming nodes..."):
            transformed.append(await task)

        return transformed

    @sequential_task()
    async def transform_node(self, node: BaseNode, **kwargs) -> BaseNode:
        entities = node.metadata.pop("entities", None)
        relationships = node.metadata.pop("relationships", None)
        claims = node.metadata.pop("claims", None)

        if entities:
            for e in entities:
                await self._upsert_entity_node(node, e)

        if relationships:
            for r in relationships:
                await self._upsert_relation_node(node, r)

        if claims:
            for c in claims:
                await self._upsert_claim_node(node, c)

        node.excluded_embed_metadata_keys = self._excluded_embed_metadata_keys
        node.excluded_llm_metadata_keys = self._excluded_llm_metadata_keys
        return node

    async def _upsert_entity_node(
        self, parent_node: BaseNode, entity: DummyEntity
    ) -> EntityNode:
        get_existing_entity = self._graph_store.get(ids=[entity.key])

        existing_entity = get_existing_entity[0] if get_existing_entity else None

        entity_values = {
            "id": entity.key,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "description": (
                (
                    f"{existing_entity.properties.get('description', '')} "
                    f"{entity.description}"
                )
                if existing_entity
                else entity.description
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
                    **{attr.name: attr.value for attr in entity.attributes},
                }
            ),
            "sources": (
                existing_entity.properties.get("sources", [])
                + [parent_node.metadata.get("doc_id", parent_node.node_id)]
                if existing_entity
                else [parent_node.metadata.get("doc_id", parent_node.node_id)]
            ),
        }

        if existing_entity:
            entity_values["description"] = await self.invoke_llm(
                type="entity",
                title=entity_values["name"],
                description=entity_values["description"],
            )

        entity_node = EntityNode(
            name=entity_values["name"],
            label=entity_values["entity_type"],
            properties={
                "sources": entity_values["sources"],
                "description": entity_values["description"],
                **entity_values["attributes"],
            },
        )

        if self._graph_store.supports_vector_queries:
            entity_node.properties[
                "embedding"
            ] = await self._embeddings.aget_text_embedding(str(entity_node))

        self._graph_store.upsert_nodes([entity_node])
        return entity_node

    async def _upsert_relation_node(
        self, parent_node: BaseNode, relation: Relationship
    ) -> Relation:
        get_existing_relation = self._graph_store.get_triplets(
            ids=[relation.key],
        )

        existing_relation = get_existing_relation[0] if get_existing_relation else None

        relation_values = {
            "id": relation.key,
            "source_entity": relation.source_entity.replace('"', " "),
            "target_entity": relation.target_entity.replace('"', " "),
            "relation_type": relation.relation_type,
            "description": (
                (
                    f"{existing_relation.properties.get('description', '')} "
                    f"{relation.description}"
                )
                if existing_relation
                else relation.description
            ),
            "strength": (
                (existing_relation.properties.get("strength", 0) + relation.strength)
                / 2
                if existing_relation
                else relation.strength
            ),
            "sources": (
                existing_relation.properties.get("sources", [])
                + [parent_node.metadata.get("doc_id", parent_node.node_id)]
                if existing_relation
                else [parent_node.metadata.get("doc_id", parent_node.node_id)]
            ),
        }

        if existing_relation:
            relation_values["description"] = await self.invoke_llm(
                type="relation",
                title=relation_values["source_entity"],
                description=relation_values["description"],
            )

        relation_node = Relation(
            label=relation_values["relation_type"],
            source_id=relation_values["source_entity"],
            target_id=relation_values["target_entity"],
            properties={
                "sources": relation_values["sources"],
                "description": relation_values["description"],
                "strength": relation_values["strength"],
            },
        )

        self._graph_store.upsert_relations([relation_node])
        return relation_node

    async def _upsert_claim_node(
        self, parent_node: BaseNode, claim: DummyClaim
    ) -> ChunkNode:
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

        self._graph_store.upsert_nodes([chunk_node])
        if rel_nodes:
            self._graph_store.upsert_relations(rel_nodes)

        return chunk_node

    def _construct_function(self):
        @ell.complex(**self.config)
        def summarize_node(
            type: Literal["entity", "relation"],
            title: str,
            description: str,
        ):
            return [
                ell.system(SUMMARIZE_NODE_PROMPT),
                ell.user(SUMMARIZE_NODE_MESSAGE.format(type=type, title=title, description=description)),
            ]

        return summarize_node
