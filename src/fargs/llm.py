import os
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


class OpenAIChatModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class OpenAIEmbeddingModels(Enum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


def get_embeddings(model: str):
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
    )
