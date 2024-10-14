import hashlib
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class DefaultClaimTypes(Enum):
    FACT = "fact"
    OPINION = "opinion"


class ClaimStatus(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    SUSPECTED = "SUSPECTED"


class DummyClaim(BaseModel):
    claim_subject: str
    claim_object: str
    claim_type: Enum
    status: ClaimStatus
    title: str
    description: str
    period: str
    sources: list[str]

    def __str__(self):
        return (
            f"{self.claim_subject} -> {self.claim_type.value} -> {self.claim_object}: "
            f"{self.title}"
        )

    @property
    def key(self) -> str:
        hash_input = (
            f"{self.claim_subject}_{self.claim_object}_{self.claim_type}_"
            f"{self.status}_{self.description}"
        )
        return hashlib.md5(hash_input.encode()).hexdigest()

    @property
    def subject_key(self) -> str:
        # Here to maintain compatibility with LlamaIndex
        return self.claim_subject.replace('"', " ")

    @property
    def object_key(self) -> str | None:
        if self.claim_object:
            # Here to maintain compatibility with LlamaIndex
            return self.claim_object.replace('"', " ")
        return None


def build_model(claim_types: Enum):
    class Claim(DummyClaim):
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
        claim_type: claim_types = Field(
            title="Type",
            description="The type of the claim.",
        )
        status: ClaimStatus = Field(
            title="Status",
            description=(
                "The status of the claim. A TRUE claim is a factually verified correct "
                "claim. A FALSE claim is a factually verified incorrect claim. "
                "A SUSPECTED claim is a claim that is not factually correct or "
                "incorrect, but we are not sure."
            ),
        )
        title: str = Field(
            title="Title",
            description=("Short, single-sentence title summarizing the claim."),
        )
        description: str = Field(
            title="Description",
            description=(
                "Detailed, single-paragraph description explaining the reasoning "
                "behind the claim, together with the related evidences and references "
                "from the original text."
            ),
        )
        period: str = Field(
            title="Period",
            description=(
                "Period when the claim was made formatted as (start_date, end_date) in "
                "YYYY-MM-DD format. "
                "If the claim was made on a single day, you may use the same date for "
                "both start_date and end_date."
            ),
        )
        sources: list[str] = Field(
            title="Sources",
            description=(
                "List of ALL quotes from the original text that support or are "
                "relevant to the claim."
            ),
        )

        @field_validator("claim_subject", "claim_object", mode="before")
        @classmethod
        def capitalize_entities(cls, value):
            return value.upper()

        @field_validator("claim_type", mode="before")
        @classmethod
        def validate_claim_type(cls, value):
            claim_type_values = [t.value.lower() for t in claim_types]
            if str(value).lower() not in claim_type_values:
                if "other" in claim_type_values:
                    return "other"

                raise ValueError(f"Invalid claim type: {value}")
            return value

        def model_dump(self, *args, **kwargs):
            data = super().model_dump(*args, **kwargs)
            data["claim_type"] = data["claim_type"].value
            data["status"] = data["status"].value
            return data

    return Claim
