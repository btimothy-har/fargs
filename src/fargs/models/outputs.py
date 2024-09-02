from pydantic import BaseModel
from pydantic import Field

from .claims import Claim
from .entities import Entity
from .entities import ResolvedEntity
from .relationships import BasicRelationship


class EntityOutput(BaseModel):
    entities: list[Entity] = Field(
        title="Entities", description="List of entities identified."
    )


class NamedResolvedEntityOutput(BaseModel):
    consolidated_entity: Entity = Field(
        title="Consolidated Entity",
        description=(
            "The consolidated entity, combining all entities with the same name."
        ),
    )
    unmatched_entities: list[Entity] = Field(
        title="Unmatched Entities",
        description="Entities that do not belong to this group.",
    )


class ResolvedEntityOutput(BaseModel):
    entities: list[ResolvedEntity] = Field(
        title="Resolved Entities",
        description=("List of resolved entities, with their aliases and descriptions."),
    )


class ExtractRelationshipsOutput(BaseModel):
    relationships: list[BasicRelationship] = Field(
        title="Relationships",
        description="List of relationships identified.",
    )


class ClaimOutput(BaseModel):
    claims: list[Claim] = Field(
        title="Claims", description="List of claims identified in Step 2."
    )
