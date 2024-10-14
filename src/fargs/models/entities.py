from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class DefaultEntityTypes(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"


class EntityAttribute(BaseModel):
    name: str = Field(
        title="Name",
        description="The name of the attribute (e.g. date of birth).",
    )
    value: str = Field(
        title="Value",
        description=(
            "The value of the attribute (e.g. 1970-01-01). "
            "Dates should be in the format YYYY-MM-DD.  "
        ),
    )


class DummyEntity(BaseModel):
    name: str
    entity_type: str
    description: str
    attributes: list[EntityAttribute]

    def __str__(self):
        return f"{self.name}: {self.description}"

    @property
    def key(self):
        # Here to maintain compatibility with LlamaIndex
        return self.name.replace('"', " ")


def build_model(entity_types: Enum):
    class Entity(DummyEntity):
        name: str = Field(
            title="Name",
            description=(
                "The name of the entity, in upper case. DO NOT use abbreviations. "
            ),
        )
        entity_type: entity_types = Field(
            title="Type",
            description=(
                "The type of the entity. Only identify entities that belong to the "
                "list of valid entity types provided."
            ),
        )
        description: str = Field(
            title="Description",
            description=(
                "Comprehensive, single-paragraph description of the "
                "entity's attributes and activities."
            ),
        )
        attributes: list[EntityAttribute] = Field(
            title="Attributes",
            description=(
                "List of attributes of the entity. "
                "Attributes are additional details or characteristics that "
                "provide context about the entity. "
                "Attributes should be relatively permanent in nature. "
                "For example, a date of birth is an attribute, but age is not. "
            ),
        )

        @field_validator("name", mode="before")
        @classmethod
        def capitalize_fields(cls, value):
            return value.upper()

        @field_validator("entity_type", mode="before")
        @classmethod
        def validate_entity_type(cls, value):
            entity_type_values = [t.value.lower() for t in entity_types]
            if str(value).lower() not in entity_type_values:
                if "other" in entity_type_values:
                    return "other"

                raise ValueError(
                    f"Entity type {value} is not in the list of valid entity types."
                )
            return str(value).lower()

        def model_dump(self, *args, **kwargs):
            data = super().model_dump(*args, **kwargs)
            data["entity_type"] = data["entity_type"].value
            return data

    return Entity
