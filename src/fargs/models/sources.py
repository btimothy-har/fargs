from hashlib import md5
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class Document(BaseModel):
    doc_id: str = Field(default_factory=uuid4)
    title: str
    text: str
    content_hash: str
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def fill_metadata(cls, data):
        data["metadata"] = data.get("metadata", {})
        for key, value in data.items():
            if key not in cls.model_fields:
                data["metadata"][key] = value
        return data

    @model_validator(mode="before")
    @classmethod
    def fill_content_hash(cls, data):
        data["content_hash"] = md5(data["text"].encode()).hexdigest()
        return data


class TextUnit(BaseModel):
    doc_id: str
    unit_num: int
    text: str
    embedding: list[float] = Field(default_factory=None)
