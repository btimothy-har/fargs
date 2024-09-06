import asyncio
from abc import ABC
from abc import abstractmethod
from datetime import UTC
from datetime import datetime

import pandas as pd

from fargs.data import ClaimsParquetData
from fargs.data import DocumentsParquetData
from fargs.data import TextUnitsParquetData
from fargs.models import Claim
from fargs.models import ClaimOutput
from fargs.models import ClaimType
from fargs.models import Document
from fargs.models import Entity
from fargs.models import EntityOutput
from fargs.models import ExtractRelationshipsOutput
from fargs.models import NamedResolvedEntityOutput
from fargs.models import Relationship
from fargs.models import ResolvedEntity
from fargs.models import ResolvedEntityOutput
from fargs.models import TextUnit
from fargs.prompts import CLAIM_EXTRACTION
from fargs.prompts import EXTRACT_ENTITIES_PROMPT
from fargs.prompts import NAMED_ENTITY_RESOLUTION
from fargs.prompts import RELATIONSHIP_EXTRACTION
from fargs.prompts import RELATIONSHIP_FILLER
from fargs.prompts import SIMILAR_ENTITY_RESOLUTION

from .base import BaseFargs
from .base import FargsConfig

RELATIONSHIP_EXTRACTION_MESSAGE = """
ENTITIES
----------
{entities_json}

TEXT
----------
{text_unit}
"""

RELATIONSHIP_FILLER_MESSAGE = """
ENTITIES
----------
{source_entity}
{target_entity}

TEXT
----------
{text_unit}
"""

CLAIM_EXTRACTION_MESSAGE = """
ENTITIES
----------
{entities_json}

TEXT
----------
{text_unit}
"""


class DocumentTasks(BaseFargs, ABC):
    def __init__(self, config: FargsConfig):
        super().__init__(config)

    @property
    @abstractmethod
    def documents(self) -> DocumentsParquetData:
        pass

    @property
    @abstractmethod
    def text_units(self) -> TextUnitsParquetData:
        pass

    @property
    @abstractmethod
    def claims(self) -> ClaimsParquetData:
        pass

    async def embed_text_unit(self, text_unit: TextUnit) -> TextUnit:
        text_unit.embedding = await self.embed_text(text_unit.text)
        return text_unit

    async def embed_entity(self, entity: ResolvedEntity) -> dict:
        entity_dict = entity.model_dump()
        entity_dict["name_embedded"] = await self.embed_text(entity.entity_name)
        entity_dict["description_embedded"] = await self.embed_text(
            entity.entity_description
        )
        return entity_dict

    async def embed_relationship(self, relationship: Relationship) -> dict:
        rls_dict = relationship.model_dump()
        rls_dict["description_embedded"] = await self.embed_text(
            relationship.relationship_description
        )
        return rls_dict

    async def embed_claims(self, claim: Claim) -> dict:
        claim_dict = claim.model_dump()
        claim_dict["description_embedded"] = await self.embed_text(
            claim.claim_description
        )
        return claim_dict

    async def build_text_units(self, document: Document) -> pd.DataFrame:
        raw_split_text = self.chunker.split_text(document.text)
        text_units = [
            TextUnit(text=text, doc_id=document.doc_id, unit_num=i)
            for i, text in enumerate(raw_split_text)
        ]
        text_units = await asyncio.gather(
            *[asyncio.create_task(self.embed_text_unit(unit)) for unit in text_units]
        )
        return pd.DataFrame([unit.model_dump() for unit in text_units])

    async def extract_entities(self, processing_unit: TextUnit) -> pd.DataFrame:
        prompt = EXTRACT_ENTITIES_PROMPT.format(
            entity_types=[t.value for t in self.config.entity_types],
            current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

        output = self.llm.with_structured_output(EntityOutput)

        llm_input = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": processing_unit.text},
        ]
        response = await self.invoke_llm(output.ainvoke, llm_input)
        entities = [entity for entity in response.entities]

        for e in entities:
            e.e_id = processing_unit.doc_id
            e.e_source_units = [f"{processing_unit.doc_id}-{processing_unit.unit_num}"]
        return pd.DataFrame([entity.model_dump() for entity in entities])

    async def resolve_entities_by_name(
        self, document: Document, processing_entities: pd.DataFrame
    ) -> pd.DataFrame:
        entities = [
            Entity(**entity) for entity in processing_entities.to_dict(orient="records")
        ]
        entity_names = set([entity.entity_name for entity in entities])

        output = self.llm.with_structured_output(NamedResolvedEntityOutput)

        async def resolve_entity(entity_name: str):
            similar_entities = [e for e in entities if e.entity_name == entity_name]

            if len(similar_entities) == 1:
                return similar_entities[0]

            llm_input = [
                {
                    "role": "system",
                    "content": NAMED_ENTITY_RESOLUTION,
                },
                {
                    "role": "user",
                    "content": ", ".join(
                        [entity.model_dump_json() for entity in similar_entities]
                    ),
                },
            ]
            resolution = await self.invoke_llm(output.ainvoke, llm_input)
            resolution.consolidated_entity.e_source_units = []
            for e in similar_entities:
                resolution.consolidated_entity.e_source_units.extend(e.e_source_units)

            for e in resolution.unmatched_entities:
                e.e_source_units = []
                for e in similar_entities:
                    e.e_source_units.extend(e.e_source_units)

            return resolution.consolidated_entity + resolution.unmatched_entities

        resolved_entities = await asyncio.gather(
            *[resolve_entity(e) for e in entity_names]
        )

        for r in resolved_entities:
            r.e_id = f"{r.entity_name}_{document.doc_id}"

        return pd.DataFrame([entity.model_dump() for entity in resolved_entities])

    async def resolve_document_entities(
        self, document: Document, processing_entities: pd.DataFrame
    ) -> pd.DataFrame:
        entities = [
            Entity(**entity) for entity in processing_entities.to_dict(orient="records")
        ]
        messages = [
            {
                "role": "system",
                "content": SIMILAR_ENTITY_RESOLUTION,
            },
            {
                "role": "user",
                "content": ", ".join(
                    [
                        entity.model_dump_json(
                            include={"entity_name", "entity_type", "entity_description"}
                        )
                        for entity in entities
                    ]
                ),
            },
        ]
        output = self.llm.with_structured_output(ResolvedEntityOutput)
        response = await self.invoke_llm(output.ainvoke, messages)

        embed_tasks = []
        for r in response.entities:
            r.e_id = f"{r.entity_name}_{document.doc_id}"
            original_entities = [
                e
                for e in entities
                if e.entity_name == r.entity_name or e.entity_name in r.aliases
            ]
            r.e_source_units = []
            for e in original_entities:
                r.e_source_units.extend(e.e_source_units)
            embed_tasks.append(asyncio.create_task(self.embed_entity(r)))

        embedded_entities = await asyncio.gather(*embed_tasks)
        return pd.DataFrame(embedded_entities)

    async def extract_relationships(
        self, processed_text_units: pd.DataFrame, processed_entities: pd.DataFrame
    ) -> pd.DataFrame:
        entities = [
            ResolvedEntity(**entity)
            for entity in processed_entities.to_dict(orient="records")
        ]

        entities_json = [
            e.model_dump_json(
                include={
                    "entity_name",
                    "entity_type",
                    "entity_description",
                }
            )
            for e in entities
        ]

        async def extract_by_text_unit(
            text_unit: TextUnit,
        ) -> pd.DataFrame:
            messages = [
                {"role": "system", "content": RELATIONSHIP_EXTRACTION},
                {
                    "role": "user",
                    "content": RELATIONSHIP_EXTRACTION_MESSAGE.format(
                        entities_json="\n".join(entities_json),
                        text_unit=text_unit.text,
                    ),
                },
            ]
            output = self.llm.with_structured_output(ExtractRelationshipsOutput)
            response = await self.invoke_llm(output.ainvoke, messages)

            output = self.llm.with_structured_output(Relationship)
            tasks = []
            for rls in response.relationships:
                try:
                    source_ent = [
                        e for e in entities if e.entity_name == rls.source_entity
                    ][0]
                    target_ent = [
                        e for e in entities if e.entity_name == rls.target_entity
                    ][0]
                except IndexError:
                    continue

                messages = [
                    {"role": "system", "content": RELATIONSHIP_FILLER},
                    {
                        "role": "user",
                        "content": RELATIONSHIP_FILLER_MESSAGE.format(
                            source_entity=source_ent.model_dump_json(
                                include={
                                    "entity_name",
                                    "entity_type",
                                    "entity_description",
                                }
                            ),
                            target_entity=target_ent.model_dump_json(
                                include={
                                    "entity_name",
                                    "entity_type",
                                    "entity_description",
                                }
                            ),
                            text_unit=text_unit.text,
                        ),
                    },
                ]
                tasks.append(
                    asyncio.create_task(self.invoke_llm(output.ainvoke, messages))
                )

            filled_relationships = await asyncio.gather(*tasks)
            for rls in filled_relationships:
                rls.r_source_units = [f"{text_unit.doc_id}-{text_unit.unit_num}"]
                try:
                    original_source_entity = [
                        e
                        for e in entities
                        if e.entity_name == rls.source_entity
                        or rls.source_entity in e.aliases
                    ][0]
                except IndexError:
                    rls.source_id = None
                else:
                    rls.source_id = original_source_entity.e_id

                try:
                    original_target_entity = [
                        e
                        for e in entities
                        if e.entity_name == rls.target_entity
                        or rls.target_entity in e.aliases
                    ][0]
                except IndexError:
                    rls.target_id = None
                else:
                    rls.target_id = original_target_entity.e_id

            return filled_relationships

        tasks = []
        for _, row in processed_text_units.iterrows():
            task = asyncio.create_task(extract_by_text_unit(TextUnit(**row.to_dict())))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        embed_relationship_tasks = []
        for list_rls in responses:
            for rls in list_rls:
                embed_relationship_tasks.append(
                    asyncio.create_task(self.embed_relationship(rls))
                )
        embedded_relationships = await asyncio.gather(*embed_relationship_tasks)
        return pd.DataFrame(embedded_relationships)

    async def extract_claims(
        self, processed_text_units: pd.DataFrame, processed_entities: pd.DataFrame
    ) -> pd.DataFrame:
        entities = [
            ResolvedEntity(**entity)
            for entity in processed_entities.to_dict(orient="records")
        ]
        entities_json = [
            e.model_dump_json(
                include={
                    "entity_name",
                    "entity_type",
                    "entity_description",
                }
            )
            for e in entities
        ]
        prompt = CLAIM_EXTRACTION.format(
            claim_types=[t.value for t in ClaimType],
            current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

        async def extract_by_text_unit(text_unit: TextUnit) -> ClaimOutput:
            output = self.llm.with_structured_output(ClaimOutput)
            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": CLAIM_EXTRACTION_MESSAGE.format(
                        entities_json=", ".join(entities_json),
                        text_unit=text_unit.text,
                    ),
                },
            ]
            response = await self.invoke_llm(output.ainvoke, messages)

            for claim in response.claims:
                claim.c_source_units = [f"{text_unit.doc_id}-{text_unit.unit_num}"]
                try:
                    original_subject = [
                        e
                        for e in entities
                        if e.entity_name == claim.claim_subject
                        or claim.claim_subject in e.aliases
                    ][0]
                except IndexError:
                    claim.subject_id = None
                else:
                    claim.subject_id = original_subject.e_id

                if claim.claim_object != "NONE":
                    try:
                        original_object = [
                            e
                            for e in entities
                            if e.entity_name == claim.claim_object
                            or claim.claim_object in e.aliases
                        ][0]
                    except IndexError:
                        claim.object_id = None
                    else:
                        claim.object_id = original_object.e_id

            return response.claims

        tasks = []
        for _, row in processed_text_units.iterrows():
            task = asyncio.create_task(extract_by_text_unit(TextUnit(**row.to_dict())))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)

        discovered_claims = [claim for response in responses for claim in response]
        tasks = [
            asyncio.create_task(self.embed_claims(claim)) for claim in discovered_claims
        ]
        embedded_claims = await asyncio.gather(*tasks)
        return pd.DataFrame(embedded_claims)
