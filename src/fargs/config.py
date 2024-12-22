import os

from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingMode
from pydantic import BaseModel
from pydantic_ai.models import KnownModelName
from pydantic_ai.models import Model
from typing_extensions import TypedDict

EMBEDDING_CONTEXT_LENGTH = 6_000
SUMMARY_CONTEXT_WINDOW = 100_000

PROCESSING_BATCH_SIZE = int(os.getenv("FARGS_BATCH_SIZE", "100"))


class FargsPrompts(BaseModel):
    claims: str = None
    entities: str = None
    relationships: str = None
    summarization: str = None


class LLMConfiguration(TypedDict):
    model: str | Model | KnownModelName = "openai:gpt-4o-mini"
    temperature: float = 0.0


class RetryConfig(TypedDict):
    tries: int = 3
    delay: int = 1
    max_delay: int = 10
    backoff: float = 1.5
    jitter: int = 1


default_embeddings = OpenAIEmbedding(
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=os.getenv("OPENAI_API_KEY"),
)

default_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=256, separator=". ")

default_graph_store = SimplePropertyGraphStore()

default_vector_store = SimpleVectorStore()
