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
    summary: str = Field(
        title="Summary",
        description=(
            "An executive summary of the community's overall structure, "
            "how its entities are related to each other, and significant information "
            "associated with its entities."
        ),
    )
    impact_severity_rating: float = Field(
        title="Impact Severity Rating",
        description=(
            "A float score between 0-10 that represents the severity of IMPACT posed "
            "by entities within the community. IMPACT is the scored importance of a "
            "community."
        ),
    )
    rating_explanation: str = Field(
        title="Rating Explanation",
        description=(
            "A single sentence explanation of the IMPACT severity rating. "
            "This explanation should be concise and directly address the impact of the "
            "community on the overall structure and importance of the community."
        ),
    )
    detailed_findings: list[str] = Field(
        title="Detailed Findings",
        description=(
            "A list of 5-10 key insights about the community. Each insight should "
            "have a short summary followed by multiple paragraphs of explanatory text "
            "grounded according to the grounding rules below. Be comprehensive."
        ),
    )
