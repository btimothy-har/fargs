import asyncio
import os
from enum import Enum

from llama_index.core.embeddings import OpenAIEmbedding
from llama_index.core.embeddings import OpenAIEmbeddingMode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from fargs import Fargs


class ClaimTypes(Enum):
    FACT = "fact"
    OPINION = "opinion"
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    DENIAL = "denial"
    CONFIRMATION = "confirmation"
    ACCUSATION = "accusation"
    PROMISE = "promise"
    WARNING = "warning"
    ANNOUNCEMENT = "announcement"


class EntityTypes(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    INDUSTRY = "industry"
    LOCATION = "location"
    LANGUAGE = "language"
    CURRENCY = "currency"
    GEOPOLITICAL_ENTITY = "geopolitical_entity"
    NORP = "nationality_or_religious_or_political_group"
    POSITION = "position"
    LEGAL = "legal_documents_or_laws_or_treaties"
    ART = "work_of_art"
    PRODUCT_OR_SERVICE = "product_or_service"
    EVENT = "event"
    INFRASTRUCTURE = "infrastructure"


embeddings = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=os.getenv("OPENAI_API_KEY"),
)

splitter = SemanticSplitterNodeParser(
    buffer_size=2,
    embed_model=embeddings,
    breakpoint_percentile_threshold=95,
)

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="neo4j",
    url="bolt://localhost:7687",
    refresh_schema=False,
)

fargs = Fargs(
    project_name="my_graph_project",
    pre_split_strategy=splitter,
    embedding_strategy=embeddings,
    graph_store=graph_store,
    extraction_llm_model={
        "model": "gpt-4o",
        "temperature": 0,
    },
    summarization_llm_model={
        "model": "gpt-4o",
        "temperature": 0,
    },
    entity_types=EntityTypes,
    claim_types=ClaimTypes,
)

documents = [
    Document(text="Hello, world!"),
]

asyncio.run(fargs.ingest(documents))
