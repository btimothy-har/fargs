from aenum import Enum
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class DefaultEntityTypes(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    INDUSTRY = "industry"
    LOCATION = "location"
    LANGUAGE = "language"
    CURRENCY = "currency"
    NUMBER = "number"
    DATETIME = "date_or_time"
    GEOPOLITICAL_ENTITY = "geopolitical_entity"
    NORP = "nationality_or_religious_or_political_group"
    LEGAL = "legal_documents_or_laws_or_treaties"
    ART = "work_of_art"
    PRODUCT_OR_SERVICE = "product_or_service"
    EVENT = "event"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class Entity(BaseModel):
    entity_name: str = Field(
        title="Entity Name",
        description=(
            "The name of the entity, in upper case. "
            "Entity names should be full and complete versions, unabbreviated. "
            "(Example: UNITED NATIONS instead of UN)"
        ),
    )
    entity_type: DefaultEntityTypes = Field(
        title="Entity Type",
        description=(
            "The type of the entity. Select from the list of valid "
            "types provided to you in your instructions. If none of the types match, "
            "you may use other."
        ),
    )
    entity_description: str = Field(
        title="Description",
        description=(
            "Comprehensive, single-paragraph description of the "
            "entity's attributes and activities."
        ),
    )

    @field_validator("entity_name", mode="before")
    @classmethod
    def capitalize_fields(cls, value):
        return value.upper()

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_entity_type(cls, value):
        if str(value).lower() not in [t.value for t in DefaultEntityTypes]:
            return DefaultEntityTypes.OTHER.value
        return str(value).lower()

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["entity_type"] = data["entity_type"].value
        return data


class ResolvedEntity(Entity):
    aliases: list[str] = Field(
        title="Aliases",
        description="List of aliases for the entity, capitalized.",
    )

    @model_validator(mode="after")
    def remove_self_alias(self):
        self.aliases = [alias for alias in self.aliases if alias != self.entity_name]
        return self
