from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class Document(BaseModel):
    doc_id: str = Field(default_factory=uuid4)
    text: str
    metadata: dict = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if key not in self.model_fields:
                self.metadata[key] = value

    def dataframe_dict(self) -> dict:
        output = self.model_dump()
        for key, value in output["metadata"].items():
            output[key] = value
        del output["metadata"]
        return output


class TextUnit(BaseModel):
    doc_id: str
    unit_num: int
    text: str
    embedding: list[float] = Field(default_factory=None)
