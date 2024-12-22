# ruff: noqa: E501

EXTRACT_ENTITIES_PROMPT = """
### YOUR GOAL ###
You will be presented with a piece of text. Your task is to identify key named entities present in the text, and any relevant attributes.

Entities are defined as an explicitly named object or concept that has a distinct identity and meaning.
Entities must be explicitly named in the text, using proper nouns.

EXAMPLES:
- The use of the word "She" is not a proper noun, and should not be identified as an entity.
- "U.S." should be identified as "United States of America".
- "NATO" should be identified as "North Atlantic Treaty Organization".
- DATES should not be identified as entities.

The date today is {current_date}. You may use this information to help identify entities, but should not be treated as an entity itself.

### INSTRUCTIONS ###
From the text provided, identify the named entities.

Only identify entities that belong to one of the following types:
{entity_types}

For each identified entity, extract the following information:
- name: Name of the entity, non-abbreviated and capitalized.
- entity_type: The type of the entity.
- description: Comprehensive description of the entity's attributes and activities.
- attributes: List of attributes of the entity.

If there are no entities to identify, use the schema flag "no_entities" to indicate that no entities were identified.
"""
