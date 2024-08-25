# ruff: noqa: E501

CLAIM_EXTRACTION = """
### YOUR GOAL ###
You will be presented with an incomplete text document, alongside with a list entities that have been identified from the broader context.
Your task is to extract and analyze claims made by the various entities.

### INSTRUCTIONS ###
STEP 1
----------
Review the entities provided to you.

STEP 2
----------
For each entity, extract all claims actively made by this entity. A passive claim should not be considered a claim.
A claim may affect or involve other entities - you shall label these as claim_object.

Not all entities may appear in the text or may make claims - you may ignore these entities.

For each claim, extract the following information:
- claim_subject: name of the entity that is subject of the claim, capitalized. This should be the entity in question.
- claim_object: name of the entity that is the object of the claim, capitalized. The object entity is the entity that either is affected by the action described in the claim. If there is no object entity, you may use NONE.
- claim_type: the type of the claim, one of: {claim_types}
- claim_status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- claim_description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references from the original text.
- claim_period: Period when the claim was made formatted as (start_date, end_date). If the claim was made on a single day, you may use the same date for both start_date and end_date.
- claim_source_text: List of **all** quotes from the original text that are relevant to the claim.

STEP 3
----------
Consolidate your output from Step 2 into the schema provided to you.

### ADDITIONAL INSTRUCTIONS ###

Dates and Times
----------
When identifying dates or times, always use the following format: YYYY-MM-DD.

If the complete date is not known, you may use your knowledge of the current date: {current_date}.
"""
