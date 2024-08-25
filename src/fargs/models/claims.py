from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class ClaimType(Enum):
    FACT = "fact"
    OPINION = "opinion"
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    DENIAL = "denial"
    CONFIRMATION = "confirmation"
    ACCUSATION = "accusation"
    PROMISE = "promise"
    WARNING = "warning"
    ANNOUNCEMENT = "announcement"
    OTHER = "other"


class ClaimStatus(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    SUSPECTED = "SUSPECTED"


class Claim(BaseModel):
    claim_subject: str = Field(
        title="Subject",
        description=(
            "Name of the entity that is the subject of the claim. "
            "The subject entity is the entity that committed the action described "
            "in the claim."
        ),
    )
    claim_object: str = Field(
        title="Object",
        description=(
            "Name of the entity that is the object of the claim. "
            "The object entity is the entity that either is affected by the action "
            "described in the claim. "
            "If there is no object entity, you may use NONE."
        ),
    )
    claim_type: ClaimType = Field(
        title="Type",
        description="The type of the claim.",
    )
    claim_status: ClaimStatus = Field(
        title="Status",
        description=(
            "The status of the claim. A TRUE claim is a factually verified correct "
            "claim. A FALSE claim is a factually verified incorrect claim. A "
            "SUSPECTED claim is a claim that is not factually correct or incorrect, "
            "but we are not sure."
        ),
    )
    claim_description: str = Field(
        title="Description",
        description=(
            "Detailed, single-paragraph description explaining the reasoning "
            "behind the claim, together with the related evidences and references "
            "from the original text."
        ),
    )
    claim_period: str = Field(
        title="Period",
        description=(
            "Period when the claim was made formatted as (start_date, end_date) in "
            "YYYY-MM-DD format. "
            "If the claim was made on a single day, you may use the same date for both "
            "start_date and end_date."
        ),
    )
    claim_source_text: list[str] = Field(
        title="Sources",
        description=(
            "List of ALL quotes from the original text that support or are relevant "
            "to the claim."
        ),
    )

    @field_validator("claim_subject", "claim_object", mode="before")
    @classmethod
    def capitalize_entities(cls, value):
        return value.upper()

    @field_validator("claim_type", mode="before")
    @classmethod
    def validate_claim_type(cls, value):
        if value not in [t.value for t in ClaimType]:
            return ClaimType.OTHER.value
        return value
