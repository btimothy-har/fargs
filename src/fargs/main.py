import json
from collections import defaultdict
from enum import Enum

import networkx as nx
from llama_index.core import PropertyGraphIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from fargs.config import FargsPrompts
from fargs.config import default_embeddings
from fargs.config import default_graph_store
from fargs.config import default_splitter
from fargs.config import default_vector_store

from .components import ClaimsExtractor
from .components import CommunitySummarizer
from .components import EntityExtractor
from .components import GraphLoader
from .components import RelationshipExtractor

INGESTION_RESTRICTED_KEYS = [
    "project_name",
    "name",
    "transformations",
    "vector_store",
]


class Fargs:
    """
    The main Fargs class.

    Provides a wrapper class to the ingestion, summarization, and search pipelines.
    """  # noqa: E501

    def __init__(
        self,
        project_name: str,
        pre_split_strategy: NodeParser = default_splitter,
        post_split_strategy: NodeParser = default_splitter,
        embedding_strategy: BaseEmbedding = default_embeddings,
        graph_store: PropertyGraphStore = default_graph_store,
        extraction_config: dict = None,
        extraction_llm_model: dict = None,
        nodes_vector_store: BasePydanticVectorStore = default_vector_store,
        summarization_config: dict = None,
        summarization_llm_model: dict = None,
        community_vector_store: BasePydanticVectorStore = default_vector_store,
        excluded_embed_metadata_keys: list[str] = None,
        excluded_llm_metadata_keys: list[str] = None,
        prompts: FargsPrompts = None,
        entity_types: Enum = None,
        claim_types: Enum = None,
        index_kwargs: dict = None,
    ):
        """
        Initialize a Fargs instance.

        Args:
            project_name (str): The name of the project.
            pre_split_strategy (NodeParser, optional): Strategy for splitting documents for extraction. Defaults to a TokenTextSplitter.
            post_split_strategy (NodeParser, optional): Strategy for splitting documents for loading. Defaults to a TokenTextSplitter.
            embedding_strategy (BaseEmbedding, optional): Strategy for embedding text. Defaults to OpenAI text-embedding-3-small.
            graph_store (PropertyGraphStore, optional): Store for the property graph. Defaults to a SimplePropertyGraphStore.
            extraction_config (dict, optional): Additional parameters to pass to the extraction pipeline. Defaults to None.
            extraction_llm_model (dict, optional): LLM model configuration for extraction. Defaults to None.
            nodes_vector_store (BasePydanticVectorStore, optional): Vector store for nodes. Defaults to a SimplePydanticVectorStore.
            summarization_config (dict, optional): Configuration for the summarization pipeline. Defaults to None.
            summarization_llm_model (dict, optional): LLM model configuration for summarization. Defaults to None.
            community_vector_store (BasePydanticVectorStore, optional): Vector store for communities. Defaults to a SimplePydanticVectorStore.
            excluded_embed_metadata_keys (list[str], optional): Keys to exclude from the embedding metadata. Defaults to None.
            excluded_llm_metadata_keys (list[str], optional): Keys to exclude from the LLM metadata. Defaults to None.
            prompts (FargsPrompts, optional): Overwrite the default prompts for various tasks. Defaults to None.
            entity_types (Enum, optional): Enumeration of entity types. Defaults to None.
            claim_types (Enum, optional): Enumeration of claim types. Defaults to None.
            index_kwargs (dict, optional): Additional parameters to pass to the index. Defaults to None.
        """  # noqa: E501
        self.project_name = project_name

        self.pre_split_strategy = pre_split_strategy
        self.post_split_strategy = post_split_strategy
        self.embedding_strategy = embedding_strategy
        self.graph_store = graph_store

        self.extraction_config = extraction_config or {}
        self.extraction_llm_model = extraction_llm_model

        self.summarization_config = summarization_config or {}
        self.summarization_llm_model = summarization_llm_model

        self.nodes_vector_store = nodes_vector_store
        self.community_vector_store = community_vector_store

        self.entity_types = entity_types
        self.claim_types = claim_types

        self.excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self.excluded_llm_metadata_keys = excluded_llm_metadata_keys

        self.index_kwargs = index_kwargs or {}
        self.index_kwargs.pop("property_graph_store", None)
        self.index_kwargs.pop("embed_model", None)
        self.index_kwargs.pop("use_async", None)

        self.prompts = prompts

        self._components = {
            "entities": EntityExtractor,
            "relationships": RelationshipExtractor,
            "claims": ClaimsExtractor,
            "communities": CommunitySummarizer,
            "graph": GraphLoader,
        }

        self._extraction_pipeline = None
        self._summarization_pipeline = None
        self._entity_community_map = None

    @property
    def extraction_pipeline(self):
        if self._extraction_pipeline is None:
            for key in INGESTION_RESTRICTED_KEYS:
                if key in self.extraction_config:
                    raise ValueError(
                        f"'{key}' should not be provided in extraction_config"
                    )

            self._extraction_pipeline = IngestionPipeline(
                name=f"{self.project_name}_graph_extraction",
                project_name=self.project_name,
                vector_store=self.nodes_vector_store,
                transformations=[
                    self.pre_split_strategy,
                    self._components["entities"](
                        prompt=self.prompts.entities if self.prompts else None,
                        entity_types=self.entity_types,
                        overwrite_config=self.extraction_llm_model,
                    ),
                    self._components["relationships"](
                        prompt=self.prompts.relationships if self.prompts else None,
                        overwrite_config=self.extraction_llm_model,
                    ),
                    self._components["claims"](
                        prompt=self.prompts.claims if self.prompts else None,
                        claim_types=self.claim_types,
                        overwrite_config=self.extraction_llm_model,
                    ),
                    self._components["graph"](
                        self.graph_store,
                        self.embedding_strategy,
                        excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                    ),
                    self.post_split_strategy,
                    self.embedding_strategy,
                ],
                **self.extraction_config,
            )
        return self._extraction_pipeline

    @property
    def summarization_pipeline(self):
        if self._summarization_pipeline is None:
            for key in INGESTION_RESTRICTED_KEYS:
                if key in self.summarization_config:
                    raise ValueError(
                        f"'{key}' should not be provided in summarization_config"
                    )

            self._summarization_pipeline = IngestionPipeline(
                name=f"{self.project_name}_graph_summarization",
                project_name=self.project_name,
                vector_store=self.community_vector_store,
                transformations=[
                    self._components["communities"](
                        prompt=self.prompts.communities if self.prompts else None,
                        overwrite_config=self.summarization_llm_model,
                    ),
                    self.embedding_strategy,
                ],
                **self.summarization_config,
            )
        return self._summarization_pipeline

    @property
    def graph_index(self):
        return PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            embed_model=self.embedding_strategy,
            **self.index_kwargs,
        )

    @property
    def nodes_index(self):
        return VectorStoreIndex.from_vector_store(
            vector_store=self.nodes_vector_store,
            embed_model=self.embedding_strategy,
            **self.index_kwargs,
        )

    @property
    def community_index(self):
        return VectorStoreIndex.from_vector_store(
            vector_store=self.community_vector_store,
            embed_model=self.embedding_strategy,
            **self.index_kwargs,
        )

    async def ingest(self, **kwargs) -> list[BaseNode] | None:
        extracted = await self.extraction_pipeline.arun(**kwargs)
        return extracted

    async def summarize(self, max_cluster_size=10) -> list[BaseNode] | None:
        from graspologic.partition import hierarchical_leiden

        nx_graph = self._create_nx_graph()
        if not nx_graph:
            return None

        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=max_cluster_size
        )
        self._entity_community_map, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        community_documents = self._transform_communities(community_info)

        community_reports = await self.summarization_pipeline.arun(
            documents=list(community_documents.values()),
            show_progress=True,
        )
        return community_reports

    async def search_nodes(self, query: str, **kwargs) -> list[NodeWithScore]:
        retriever = self.nodes_index.as_retriever(**kwargs)

        return await retriever.aretrieve(query)

    async def local_search(self, query: str, **kwargs) -> list[NodeWithScore]:
        include_text = kwargs.pop("include_text", False)
        include_properties = kwargs.pop("include_properties", False)
        path_depth = kwargs.pop("path_depth", 2)

        retriever = self.graph_index.as_retriever(
            include_text=include_text,
            include_properties=include_properties,
            path_depth=path_depth,
            **kwargs,
        )

        return await retriever.aretrieve(query)

    async def global_search(self, query: str, **kwargs) -> list[NodeWithScore]:
        retriever = self.community_index.as_retriever(**kwargs)

        retrieved_communities = await retriever.aretrieve(query)

        for community in retrieved_communities:
            entity_names = community.metadata.get("entities", [])
            entities = await self.graph_store.aget(ids=entity_names)
            community.metadata["entities"] = entities

        return retrieved_communities

    def _create_nx_graph(self) -> nx.Graph:
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.graph_store.get_triplets()

        if not triplets:
            return None

        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.id)
            nx_graph.add_node(entity2.id)
            nx_graph.add_edge(
                entity1.id,
                entity2.id,
                relationship=relation.label,
                description=relation.properties["description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters) -> tuple[dict, dict]:
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = {
                        "relationship": (
                            f"{node} -> {edge_data['relationship']} -> {neighbor}"
                        ),
                        "description": edge_data["description"],
                        "_source": node,
                        "_target": neighbor,
                    }
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}
        return dict(entity_info), dict(community_info)

    def _transform_communities(self, community_info) -> list[BaseNode]:
        communities = dict()

        for community_id, details in community_info.items():
            detail_data = [
                {
                    "relationship": d["relationship"],
                    "description": d["description"],
                }
                for d in details
            ]
            sources = set(d["_source"] for d in details)
            targets = set(d["_target"] for d in details)
            entities = sorted(list(set(sources.union(targets))))

            raw_entities = self.graph_store.get(ids=entities)
            source_docs = sum(
                (e.properties.get("references_", []) for e in raw_entities), []
            )

            communities[community_id] = Document(
                text="\n".join([json.dumps(d) for d in detail_data]),
                metadata={
                    "community_id": community_id,
                    "entities": entities,
                    "sources": source_docs,
                },
            )

        return communities
