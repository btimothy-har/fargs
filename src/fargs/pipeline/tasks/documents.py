import asyncio
from datetime import UTC
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fargs.models import ClaimOutput
from fargs.models import ClaimType
from fargs.models import Entity
from fargs.models import EntityOutput
from fargs.models import ExtractRelationshipsOutput
from fargs.models import NamedResolvedEntityOutput
from fargs.models import Relationship
from fargs.models import ResolvedEntity
from fargs.models import ResolvedEntityOutput
from fargs.models import TextUnit
from fargs.pipeline.prompts import CLAIM_EXTRACTION
from fargs.pipeline.prompts import EXTRACT_ENTITIES_PROMPT
from fargs.pipeline.prompts import NAMED_ENTITY_RESOLUTION
from fargs.pipeline.prompts import RELATIONSHIP_EXTRACTION
from fargs.pipeline.prompts import RELATIONSHIP_FILLER
from fargs.pipeline.prompts import SIMILAR_ENTITY_RESOLUTION


async def _embed_entity(pipeline, entity) -> dict:
    entity_dict = entity.model_dump()
    entity_dict["name_embedded"] = await pipeline._embed_text(entity.entity_name)
    entity_dict["description_embedded"] = await pipeline._embed_text(
        entity.entity_description
    )
    return entity_dict


async def _embed_relationships(pipeline, relationship) -> dict:
    relationship_dict = relationship.model_dump()
    relationship_dict["description_embedded"] = await pipeline._embed_text(
        relationship.relationship_description
    )
    return relationship_dict


async def _embed_claims(pipeline, claim) -> dict:
    claim_dict = claim.model_dump()
    claim_dict["description_embedded"] = await pipeline._embed_text(
        claim.claim_description
    )
    return claim_dict


async def _extract_relationship(pipeline, entity_df, text_unit) -> pd.DataFrame:
    output = pipeline.llm.with_structured_output(ExtractRelationshipsOutput)

    entities_json = entity_df[
        ["entity_name", "entity_type", "entity_description"]
    ].to_json(orient="records")

    messages = [
        {"role": "system", "content": RELATIONSHIP_EXTRACTION},
        {
            "role": "user",
            "content": f"""
ENTITIES
----------
{entities_json}

TEXT
----------
{text_unit['text']}
                    """,
        },
    ]

    response = await pipeline._invoke_llm(output.ainvoke, messages)

    tasks = []
    output = pipeline.llm.with_structured_output(Relationship)

    for rls in response.relationships:
        try:
            source_ent = entity_df[
                entity_df["entity_name"].str.lower() == rls.source_entity.lower()
            ].iloc[0]
            target_ent = entity_df[
                entity_df["entity_name"].str.lower() == rls.target_entity.lower()
            ].iloc[0]
        except IndexError:
            continue

        messages = [
            {"role": "system", "content": RELATIONSHIP_FILLER},
            {
                "role": "user",
                "content": f"""
ENTITIES
----------
{source_ent.to_dict()}
{target_ent.to_dict()}

TEXT
----------
{text_unit['text']}
                    """,
            },
        ]
        tasks.append(
            asyncio.create_task(pipeline._invoke_llm(output.ainvoke, messages))
        )

    responses = await asyncio.gather(*tasks)
    return responses


async def _resolve_entity_by_similarity(
    pipeline, base_df, index, tfidf, matrix, entity
):
    embed_description = entity["description_embedded"]
    _, idx = index.search(np.array([embed_description]).astype(np.float32), 20)
    desc_idx = idx.squeeze()

    # lexical match for names
    embed_name_sparse = tfidf.transform([entity["entity_name"]])
    name_similarities = cosine_similarity(embed_name_sparse, matrix).ravel()
    name_idx = np.argsort(name_similarities)[-10:][::-1]

    similar_idx = np.concatenate([name_idx, desc_idx])

    entities_to_resolve = [
        ResolvedEntity(**base_df.iloc[i].to_dict()) for i in similar_idx
    ] + [ResolvedEntity(**entity.to_dict())]

    messages = [
        {
            "role": "system",
            "content": SIMILAR_ENTITY_RESOLUTION,
        },
        {
            "role": "user",
            "content": str(
                [entity.model_dump_json() for entity in entities_to_resolve]
            ),
        },
    ]

    output = pipeline.llm.with_structured_output(ResolvedEntityOutput)

    response = await pipeline._invoke_llm(output.ainvoke, messages)
    return similar_idx, response.entities


async def build_text_units(pipeline, progress, document):
    progress.next_step("Splitting Document", 1)
    raw_split_text = pipeline._split_text(document.text)

    progress.update()

    tasks = [asyncio.create_task(pipeline._embed_text(text)) for text in raw_split_text]
    progress.next_step("Embedding Text Units", len(tasks))
    embedded_text = []
    for task in asyncio.as_completed(tasks):
        embed_text = await task
        embedded_text.append(embed_text)
        progress.update()

    text_units = [
        TextUnit(
            doc_id=document.doc_id,
            unit_num=i,
            text=raw_split_text[i],
            embedding=embedded_text[i],
        )
        for i in range(len(raw_split_text))
    ]

    return pd.DataFrame([unit.model_dump() for unit in text_units])


async def extract_entities(pipeline, progress, document_df) -> pd.DataFrame:
    text_units = [TextUnit(**row.to_dict()) for _, row in document_df.iterrows()]

    prompt = EXTRACT_ENTITIES_PROMPT.format(
        entity_types=[t.value for t in pipeline.entity_types],
        current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
    )

    output = pipeline.llm.with_structured_output(EntityOutput)

    tasks = []
    for unit in text_units:
        llm_input = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": unit.text},
        ]
        task = asyncio.create_task(pipeline._invoke_llm(output.ainvoke, llm_input))
        tasks.append(task)

    progress.next_step("Extract Entities", len(tasks))
    entities = []
    for task in asyncio.as_completed(tasks):
        response = await task
        entities.extend(response.entities)
        progress.update()

    return pd.DataFrame([entity.model_dump() for entity in entities])


async def resolve_entities_by_name(pipeline, progress, entity_df) -> pd.DataFrame:
    output = pipeline.llm.with_structured_output(NamedResolvedEntityOutput)

    consolidated_entities = []
    unmatched_entities = []
    entity_names = set(entity_df["entity_name"])

    tasks = []
    for entity_name in entity_names:
        similar_entities = [
            Entity(**entity)
            for entity in entity_df[entity_df["entity_name"] == entity_name].to_dict(
                orient="records"
            )
        ]

        if len(similar_entities) == 1:
            consolidated_entities.append(similar_entities[0])
            continue

        llm_input = [
            {
                "role": "system",
                "content": NAMED_ENTITY_RESOLUTION,
            },
            {
                "role": "user",
                "content": str(
                    [entity.model_dump_json() for entity in similar_entities]
                ),
            },
        ]
        task = asyncio.create_task(pipeline._invoke_llm(output.ainvoke, llm_input))
        tasks.append(task)

    progress.next_step("Resolve Entities", len(tasks))
    responses = []
    for task in asyncio.as_completed(tasks):
        response = await task
        responses.append(response)
        progress.update()

    for response in responses:
        consolidated_entities.append(response.consolidated_entity)
        unmatched_entities.extend(response.unmatched_entities)

    entities = consolidated_entities + unmatched_entities
    return pd.DataFrame([entity.model_dump() for entity in entities])


async def resolve_document_entities(pipeline, progress, entity_df) -> pd.DataFrame:
    progress.next_step("Resolve Entities", 1)
    output = pipeline.llm.with_structured_output(ResolvedEntityOutput)
    messages = [
        {
            "role": "system",
            "content": SIMILAR_ENTITY_RESOLUTION,
        },
        {
            "role": "user",
            "content": entity_df[
                ["entity_name", "entity_type", "entity_description"]
            ].to_json(orient="records"),
        },
    ]

    response = await pipeline._invoke_llm(output.ainvoke, messages)

    tasks = [
        asyncio.create_task(_embed_entity(pipeline, entity))
        for entity in response.entities
    ]
    progress.next_step("Embed Entities", len(tasks))
    embedded_entities = []
    for task in asyncio.as_completed(tasks):
        embed_entity = await task
        embedded_entities.append(embed_entity)

    return pd.DataFrame(embedded_entities)


async def resolve_global_entities(
    pipeline, progress, entity_df
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_df = pipeline.df_entities.copy()

    index = faiss.IndexFlatL2(1024)
    index.add(np.array(base_df["description_embedded"].tolist()).astype(np.float32))

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(base_df["entity_name"].tolist())

    tasks = []
    for _, entity in entity_df.iterrows():
        task = asyncio.create_task(
            _resolve_entity_by_similarity(
                pipeline, base_df, index, tfidf, matrix, entity
            )
        )
        tasks.append(task)

    progress.next_step("Resolving Global Entities", len(tasks))
    resolutions = []
    for task in asyncio.as_completed(tasks):
        resolution = await task
        resolutions.append(resolution)
        progress.update()

    r_idx = np.concatenate([res[0] for res in resolutions])
    r_idx_dedupe = np.unique(r_idx[r_idx >= 0])

    base_df = base_df.drop(r_idx_dedupe)

    r_entities = [entity for res in resolutions for entity in res[1]]
    tasks = [
        asyncio.create_task(_embed_entity(pipeline, entity)) for entity in r_entities
    ]
    progress.next_step("Embedding Global Entities", len(tasks))
    embed_entities = []
    for task in asyncio.as_completed(tasks):
        embed_entity = await task
        embed_entities.append(embed_entity)
        progress.update()

    embedded_df = pd.DataFrame(embed_entities)
    base_df = pd.concat([base_df, embedded_df], ignore_index=True)
    base_df = base_df.drop_duplicates(subset=["entity_name"], keep="last")
    base_df = base_df.reset_index(drop=True)

    return base_df


async def extract_relationships(
    pipeline, progress, textunit_df, entity_df
) -> pd.DataFrame:
    tasks = []
    for _, row in textunit_df.iterrows():
        task = asyncio.create_task(_extract_relationship(pipeline, entity_df, row))
        tasks.append(task)

    progress.next_step("Extract Relationships", len(tasks))
    responses = []
    for task in asyncio.as_completed(tasks):
        response = await task
        responses.append(response)
        progress.update()

    discovered_relationships = [
        relationship for response in responses for relationship in response
    ]

    tasks = [
        asyncio.create_task(_embed_relationships(pipeline, relationship))
        for relationship in discovered_relationships
    ]
    progress.next_step("Embed Relationships", len(tasks))
    embed_relationships = []
    for task in asyncio.as_completed(tasks):
        embed_relationship = await task
        embed_relationships.append(embed_relationship)
        progress.update()

    return pd.DataFrame(embed_relationships)


async def extract_claims(pipeline, progress, textunit_df, entity_df) -> pd.DataFrame:
    prompt = CLAIM_EXTRACTION.format(
        claim_types=[t.value for t in ClaimType],
        current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
    )

    output = pipeline.llm.with_structured_output(ClaimOutput)

    tasks = []
    for _, row in textunit_df.iterrows():
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"""
ENTITIES
----------
{entity_df[
    ["entity_name", "entity_type", "entity_description"]
].to_json(orient="records")}

TEXT
----------
{row['text']}
        """,
            },
        ]
        task = asyncio.create_task(pipeline._invoke_llm(output.ainvoke, messages))
        tasks.append(task)

    progress.next_step("Extract Claims", len(tasks))
    responses = []
    for task in asyncio.as_completed(tasks):
        response = await task
        responses.append(response)
        progress.update()

    discovered_claims = [claim for response in responses for claim in response.claims]

    tasks = [
        asyncio.create_task(_embed_claims(pipeline, claim))
        for claim in discovered_claims
    ]
    progress.next_step("Embed Claims", len(tasks))
    embed_claims = []
    for task in asyncio.as_completed(tasks):
        embed_claim = await task
        embed_claims.append(embed_claim)
        progress.update()

    return pd.DataFrame(embed_claims)
