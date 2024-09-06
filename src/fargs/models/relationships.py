from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class BasicRelationship(BaseModel):
    r_source_units: list[Any] | None = None
    source_entity: str = Field(title="Source", description="The source entity.")
    source_id: str | None = None
    target_entity: str = Field(title="Target", description="The target entity.")
    target_id: str | None = None

    @field_validator("source_entity", "target_entity", mode="before")
    @classmethod
    def capitalize_entity(cls, value):
        return value.upper()


class Relationship(BasicRelationship):
    relationship_description: str = Field(
        title="Description",
        description=(
            "Single-paragraph description of the relationship between the source and "
            "target entities."
        ),
    )
    relationship_strength: float = Field(
        title="Strength",
        description=(
            "A numeric float in 2 decimal places from 0.0 to 1.0 indicating the "
            "strength of the relationship."
        ),
    )

    def __str__(self):
        return (
            f"{self.source_entity} > {self.target_entity}: "
            f"{self.relationship_description}"
        )
