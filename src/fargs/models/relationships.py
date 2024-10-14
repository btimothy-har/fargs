from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class Relationship(BaseModel):
    source_entity: str = Field(title="Source", description="The source entity.")
    target_entity: str = Field(title="Target", description="The target entity.")

    relation_type: str = Field(
        title="Type",
        description="The type of relationship between the source and target entities.",
    )

    description: str = Field(
        title="Description",
        description=(
            "Single-paragraph description of the relationship between the source and "
            "target entities."
        ),
    )
    strength: float = Field(
        title="Strength",
        description=(
            "A numeric float in 2 decimal places from 0.0 to 1.0 indicating the "
            "strength of the relationship."
        ),
    )

    def __str__(self):
        return f"{self.source_entity} > {self.relation_type} > {self.target_entity}"

    @property
    def key(self):
        return (
            f"{self.source_entity.replace('"', " ")}_"
            f"{self.relation_type}_"
            f"{self.target_entity.replace('"', " ")}"
        )

    @field_validator("source_entity", "target_entity", mode="before")
    @classmethod
    def capitalize_entity(cls, value):
        return value.upper()
