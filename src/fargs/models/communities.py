from pydantic import BaseModel
from pydantic import Field


class CommunityReport(BaseModel):
    title: str = Field(
        title="Title",
        description=(
            "Community's name that represents its key entities - title should be short "
            "but specific. When possible, include representative named entities in the "
            "title."
        ),
    )
    description: str = Field(
        title="Description",
        description=(
            "A comprehensive description of the community's overall structure, "
            "how its entities are related to each other, and significant information "
            "associated with its entities."
        ),
    )
    impact_severity_rating: float = Field(
        title="Impact Severity Rating",
        description=(
            "A float score between 0-10 that represents the severity of IMPACT posed "
            "by entities within the community. The higher the score, the greater the  "
            "impact of entities on the community."
        ),
    )
    rating_explanation: str = Field(
        title="Rating Explanation",
        description=(
            "A single sentence explanation of the IMPACT severity rating. "
            "This explanation should be concise and directly address how individual "
            "entities within the community contribute to the community. "
        ),
    )
