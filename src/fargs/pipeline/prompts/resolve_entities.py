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
