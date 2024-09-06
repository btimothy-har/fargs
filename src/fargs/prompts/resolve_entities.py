# ruff: noqa: E501

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
Some of these entities might be synonyms of each other, while others are completely different entities.

Your task is to consolidate the synonyms into a single entity, while keeping the completely different entities separate.
- Synonyms MUST refer to the same object, but can have different names. EXAMPLE: the UN and United Nations are synonyms, but August 2023 (MONTH) and 13 August 2023 (DAY) are NOT.
- Names of people should always be treated as separate entities, even if they have similar names.
- DO NOT create new entities or new aliases. ONLY use the entities provided to you.

### INSTRUCTIONS ###
STEP 1
----------
Review the entities provided to you.

STEP 2
----------
Merge any synonyms into a single entity. In doing so:
- If the synonyms have been classified differently by type, determine what the best Entity Type should be and re-classify it.
- Summarize the descriptions of the synonyms into a single description.
- Include the names of the synonyms you merged as aliases of the new entity.

Separate the non-synonyms as standalone entities.

STEP 3
----------
Provide your output in the schema provided to you.

DO NOT include the name of the entity as an alias.
DO NOT include aliases in the description.

EXAMPLE 1
----------
Input: European Union, EU, European Community
Output: European Union (alias: EU), European Community

EXAMPLE 2
----------
Input: USA, United States, United States of America, North America, South America
Output: United States of America (alias: USA, United States), North America, South America

EXAMPLE 3
----------
Input: Microsoft, Microsoft Corporation, Microsoft Inc., MSFT, MS
Output: Microsoft Corporation (alias: Microsoft, Microsoft Inc., MSFT, MS)

EXAMPLE 4
----------
Input: Singapore, Singapore City, The Lion City, SGP, SG
Output: Singapore (alias: Singapore City, The Lion City, SGP, SG)

EXAMPLE 5
----------
Input: August 2023, 13 August 2023, August 9 2023, 9 August 2023
Output: August 2023, 13 August 2023, August 9 2023 (alias: 9 August 2023)

EXAMPLE 6
----------
Input: Barack Obama, Obama, Michelle Obama
Output: Barack Obama (alias: Obama), Michelle Obama
"""
