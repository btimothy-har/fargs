# ruff: noqa: E501

EXTRACT_RELATIONSHIPS_PROMPT = """
### YOUR GOAL ###
You will be presented with a piece of text, alongside with a list of entities identified in the text.
Your task is to identify all pairs of (source_entity, target_entity) that are *clearly related* to each other, as described in the text.

Relationships are defined as a connection between two entities that are explicitly mentioned in the text.

### INSTRUCTIONS ###
From the text provided and the entities identified, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.

Not all entities may appear in the text or may be related.
You are ONLY to extract the entities that are clearly related to each other AND are described in the text.

For each pair of related entities, extract the following information formatted in this schema:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relation_type: type of relationship between the source and target entities (e.g. "is_member_of", "belongs_to", "capital_of" etc.). Relation_types should be as specific as possible.
- description: explanation of the relationship between the source and target entities
- strength: a numeric float in 2 decimal places from 0.0 to 1.0 indicating the strength of the relationship

Your response should ONLY contain JSON, following this schema:
{output_schema}
"""
