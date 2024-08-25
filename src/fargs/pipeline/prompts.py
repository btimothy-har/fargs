# ruff: noqa: E501

ENTITY_RELATIONSHIP_EXTRACTION = """
### YOUR GOAL ###
You will be presented with an incomplete text document from an article.
Your task is to identify key entities present in the text, and the relationships between them.

### INSTRUCTIONS ###

STEP 1
----------
Review the text provided to you.

STEP 2
----------
Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized. If the entity is a date, ensure to include the year in the format MMM YYYY. If the year is not known, use the current year.
- entity_type: One of the following types: {entity_types}
- entity_description: Comprehensive description of the entity's attributes and activities

STEP 3
----------
Consolidate your output from Step 2 into the schema provided to you.

### ADDITIONAL INFORMATION ###

The Current Date
----------
It is currently {current_date}. The current date should not be treated as an entity.
"""


NAMED_ENTITY_RESOLUTION = """
### YOUR GOAL ###
You will be presented with a list of entities that were previously identified as present in a body of text.
These entities are all named the same, but have different descriptions.

Your task is to create a single entity that combines all of these entities with a combined description.

### INSTRUCTIONS ###
STEP 1
----------
Review the entities provided to you. These entities are all named the same, with slightly different descriptions.

STEP 2
----------
Identify if any entities are clearly different from the others, despite having the same name.
If entities share similar names and descriptions, but differ in Entity Type, determine what the best Entity Type should be and consolidate them.

If the entities cannot be consolidated, separate them out as unmatched entities.

STEP 3
----------
Summarize the remaining entities into a single entity.

STEP 4
----------
Provide your output in the schema provided to you.
"""


SIMILAR_ENTITY_RESOLUTION = """
### YOUR GOAL ###
You will be presented with a list of entities, expressed as dictionaries, that were previously identified as present in a body of text.
Some of these entities are synonyms of each other, while others are completely different entities.

Your task is to consolidate the synonyms into a single entity, while keeping the completely different entities separate.

### INSTRUCTIONS ###
STEP 1
----------
Review the entities provided to you.

STEP 2
----------
Consolidate the entities by removing the synonyms and aliases, combining them into a single entity. In doing so:
- If the entities have been classified differently by type, determine what the best Entity Type should be and consolidate them.
- Summarize the descriptions of the entities into a single description.
- Include all aliases of the entities in the consolidated description. The aliases should be drawn from the original entities, DO NOT make up new aliases.

For entities that are clearly different, separate them out as standalone entities.

STEP 3
----------
Provide your output in the schema provided to you.

DO NOT include the name of the entity as an alias.
DO NOT include aliases in the description.
"""


RELATIONSHIP_EXTRACTION = """
### YOUR GOAL ###
You will be presented with an incomplete text document, alongside with a list entities that have been identified from the broader context.
Your task is to identify all pairs of (source_entity, target_entity) that are *clearly related* to each other, as described in the text.

### INSTRUCTIONS ###

STEP 1
----------
Review the entities provided to you.

STEP 2
----------
From the entities reviewed in Step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.

Not all entities may appear in the text or may be related.
You are ONLY to extract the entities that are clearly related to each other AND are described in the text.

For each pair of related entities, extract the following information formatted in this schema:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation of the relationship between the source and target entities
- relationship_strength: a numeric float in 2 decimal places from 0.0 to 1.0 indicating the strength of the relationship

STEP 3
----------
Consolidate your output from Step 2 into the schema provided to you.
"""


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
