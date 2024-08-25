# ruff: noqa: E501

EXTRACT_ENTITIES_PROMPT = """
### YOUR GOAL ###
You will be presented with an incomplete text document from an article.
Your task is to identify key entities present in the text, and the relationships between them.

Entities are defined as an object or concept that is explicitly named in the text.

The date today is {current_date}. You may use this information to help identify entities, but should not be treated as an entity itself.

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
"""
