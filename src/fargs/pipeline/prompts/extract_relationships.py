# ruff: noqa: E501

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
