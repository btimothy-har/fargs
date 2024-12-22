# ruff: noqa: E501

EXTRACT_CLAIMS_PROMPT = """
### YOUR GOAL ###
You will be presented with a piece of text, alongside with a list of entities identified in the text.
Your task is to analyze the text and extract claims made by the various entities.

Claims are defined as statements made by an entity that assert something about the world (or another entity).

The date today is {current_date}. You may use this information to help identify the periods related to the claims.

### INSTRUCTIONS ###
From the text provided, extract all claims actively made by these entities. A passive claim should not be considered a claim.

Not all entities may appear in the text or may make claims - you may ignore these entities.

Only identify claims that are of one of the following types:
{claim_types}

For each claim, extract the following information:
- claim_subject: name of the entity that is subject of the claim, capitalized. This should be the entity in question.
- claim_object: name of the entity that is the object of the claim, capitalized. The object entity is the entity that either is affected by the action described in the claim. If there is no object entity, you may use NONE.
- claim_type: the type of the claim.
- status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- title: Short, single-sentence title summarizing the claim.
- description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references from the original text.
- period: Period when the claim was made formatted as (start_date, end_date). If the claim was made on a single day, you may use the same date for both start_date and end_date.
- sources: List of **all** quotes from the original text that are relevant to the claim.
"""
