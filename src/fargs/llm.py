import os
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from tiktoken import encoding_for_model

MAX_TOKENS_BY_MODEL = {"gpt-4o": 4_096, "gpt-4o-mini": 16_384}


class OpenAIChatModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class OpenAIEmbeddingModels(Enum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


def get_embedding_client(model: str):
    if model not in [e.value for e in OpenAIEmbeddingModels]:
        raise ValueError(f"Invalid embedding model: {model}")

    return OpenAIEmbeddings(
        model=model,
        dimensions=1024,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_chat_client(model: str):
    if model not in [e.value for e in OpenAIChatModels]:
        raise ValueError(f"Invalid chat model: {model}")

    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=MAX_TOKENS_BY_MODEL[model],
    )


def get_chunk_strategy(**kwargs: str):
    encoder = encoding_for_model(kwargs["model"])
    chunker = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoder.name,
        chunk_size=kwargs["chunk_size"],
        chunk_overlap=kwargs["chunk_overlap"],
    )
    return chunker
