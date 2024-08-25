import asyncio
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from enum import Enum
from typing import Any

import faiss
import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from pydantic import Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tiktoken import encoding_for_model

from fargs.llm import OpenAIChatModels
from fargs.llm import OpenAIEmbeddingModels
from fargs.llm import get_chat_client
from fargs.llm import get_embeddings
from fargs.models import Claim
from fargs.models import ClaimType
from fargs.models import DefaultEntityTypes
from fargs.models import Document
from fargs.models import Entity
from fargs.models import Relationship
from fargs.models import ResolvedEntity
from fargs.models import TextUnit

from .prompts import CLAIM_EXTRACTION
from .prompts import EXTRACT_ENTITIES_PROMPT
from .prompts import NAMED_ENTITY_RESOLUTION
from .prompts import RELATIONSHIP_EXTRACTION
from .prompts import SIMILAR_ENTITY_RESOLUTION


class GraphPipeline:
    def __init__(
        self,
        chat_model: str = OpenAIChatModels.GPT_4O_MINI.value,
        embedding_model: str = OpenAIEmbeddingModels.TEXT_EMBEDDING_3_SMALL.value,
        token_limit: int = 1_000_000,
        entity_types: Enum = DefaultEntityTypes,
        df_documents: pd.DataFrame = None,
        df_text_units: pd.DataFrame = None,
        df_entities: pd.DataFrame = None,
        df_relationships: pd.DataFrame = None,
        df_claims: pd.DataFrame = None,
    ):
        self.llm = get_chat_client(chat_model)
        self.embedding = get_embeddings(embedding_model)
        self.encoder = encoding_for_model(chat_model)
        self.limiter = AsyncLimiter(token_limit)
        self.chunker = SemanticChunker(
            embeddings=self.embedding,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=0.5,
        )
        self.entity_types = entity_types

        self.df_documents = df_documents
        self.df_text_units = df_text_units
        self.df_entities = df_entities
        self.df_relationships = df_relationships
        self.df_claims = df_claims

    async def ingest_document(self, document_dict: dict):
        document = Document(
            doc_id=str(document_dict["id"]),
            **document_dict,
        )

        text_units = await _build_text_units(self, document)
        entities = await _extract_entities(self, text_units)
        entities = await _resolve_entities_by_name(self, entities)
        entities = await _resolve_document_entities(self, entities)

        if self.df_entities is not None:
            self.df_entities["name_embedded"] = await asyncio.gather(
                *[self._embed_text(name) for name in self.df_entities["entity_name"]]
            )
            self.df_entities["description_embedded"] = await asyncio.gather(
                *[
                    self._embed_text(desc)
                    for desc in self.df_entities["entity_description"]
                ]
            )
            self.df_entities = await _resolve_global_entities(self, entities)

        relationships = await _extract_relationships(self, text_units, entities)
        claims = await _extract_claims(self, text_units, entities)

        dataframes = {
            "text_units": text_units,
            "entities": self.df_entities,
            "relationships": relationships,
            "claims": claims,
        }

        for name, df in dataframes.items():
            if df is not None and not df.empty:
                filename = f"{name}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")
            else:
                print(f"Skipped {name}.csv (DataFrame is None or empty)")

        if entities is not None and not entities.empty:
            filename = "entities"
            entities.to_parquet(f"{filename}.parquet", index=False)
            entities.to_csv(f"{filename}.csv", index=False)
            print(f"Saved {filename}")
        else:
            print("Skipped entities.parquet (DataFrame is None or empty)")

    def _encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def _split_text(self, text: str) -> list[str]:
        return self.chunker.split_text(text)

    async def _embed_text(self, text: str) -> list[float]:
        input_tokens = self._encode(text)

        await self.limiter.acquire(len(input_tokens))
        embeddings = await self.embedding.aembed_query(text)
        return embeddings

    async def _invoke_llm(self, task: Callable, messages: list[Any]):
        input_tokens = len(self._encode(str(messages)))

        await self.limiter.acquire(input_tokens)
        response = await task(messages)
        return response


# # async def _process_document(pipeline:GraphPipeline, document_dict: dict):
#     document = Document(**document_dict)
#     doc_df = pd.DataFrame([document.dataframe_dict()])

#     if pipeline.df_documents is None:
#         pipeline.df_documents = doc_df
#     else:
#         self.df_documents = pd.concat(
#             [self.df_documents, doc_df],
#             ignore_index=True,
#         )

#     self.df_documents = self.df_documents.drop_duplicates(
#         subset=["doc_id"], keep="last"
#     )


async def _build_text_units(pipeline: GraphPipeline, document: Document):
    raw_split_text = pipeline._split_text(document.text)
    embedded_text = await asyncio.gather(
        *(pipeline._embed_text(text) for text in raw_split_text)
    )
    text_units = [
        TextUnit(
            doc_id=document.doc_id,
            unit_num=i,
            text=text,
            embedding=embedding,
        )
        for i, (text, embedding) in enumerate(
            zip(raw_split_text, embedded_text, strict=True)
        )
    ]

    return pd.DataFrame([unit.model_dump() for unit in text_units])

    # if self.df_text_units is None:
    #     self.df_text_units = units_df
    # else:
    #     self.df_text_units = self.df_text_units[
    #         self.df_text_units["doc_id"] != document.doc_id
    #     ]
    #     self.df_text_units = pd.concat(
    #         [self.df_text_units, units_df],
    #         ignore_index=True,
    #     )


class EntityOutput(BaseModel):
    entities: list[Entity] = Field(
        title="Entities", description="List of entities identified."
    )


class NamedResolvedEntityOutput(BaseModel):
    consolidated_entity: Entity = Field(
        title="Consolidated Entity",
        description=(
            "The consolidated entity, combining all entities with the same name."
        ),
    )
    unmatched_entities: list[Entity] = Field(
        title="Unmatched Entities",
        description="Entities that do not belong to this group.",
    )


class ResolvedEntityOutput(BaseModel):
    entities: list[ResolvedEntity] = Field(
        title="Resolved Entities",
        description=("List of resolved entities, with their aliases and descriptions."),
    )


async def _extract_entities(
    pipeline: GraphPipeline, document_df: pd.DataFrame
) -> pd.DataFrame:
    text_units = [TextUnit(**row.to_dict()) for _, row in document_df.iterrows()]

    prompt = EXTRACT_ENTITIES_PROMPT.format(
        entity_types=[t.value for t in pipeline.entity_types],
        current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
    )

    output = pipeline.llm.with_structured_output(EntityOutput)

    tasks = [
        pipeline._invoke_llm(
            output.ainvoke,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": unit.text},
            ],
        )
        for unit in text_units
    ]
    responses = await asyncio.gather(*tasks)

    entities = [e for response in responses for e in response.entities]

    return pd.DataFrame([entity.model_dump() for entity in entities])


async def _resolve_entities_by_name(
    pipeline: GraphPipeline, entity_df: pd.DataFrame
) -> pd.DataFrame:
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

        tasks.append(
            pipeline._invoke_llm(
                output.ainvoke,
                [
                    {"role": "system", "content": NAMED_ENTITY_RESOLUTION},
                    {
                        "role": "user",
                        "content": str(
                            [entity.model_dump_json() for entity in similar_entities]
                        ),
                    },
                ],
            )
        )

    responses = await asyncio.gather(*tasks)

    for response in responses:
        consolidated_entities.append(response.consolidated_entity)
        unmatched_entities.extend(response.unmatched_entities)

    entities = consolidated_entities + unmatched_entities
    return pd.DataFrame([entity.model_dump() for entity in entities])


async def _resolve_document_entities(
    pipeline: GraphPipeline, entity_df: pd.DataFrame
) -> pd.DataFrame:
    output = pipeline.llm.with_structured_output(ResolvedEntityOutput)

    messages = [
        {
            "role": "system",
            "content": SIMILAR_ENTITY_RESOLUTION,
        },
        {
            "role": "user",
            "content": entity_df.to_json(orient="records"),
        },
    ]

    response = await pipeline._invoke_llm(output.ainvoke, messages)
    return pd.DataFrame([entity.model_dump() for entity in response.entities])


async def _resolve_global_entities(
    pipeline: GraphPipeline, entity_df: pd.DataFrame
) -> pd.DataFrame:
    output = pipeline.llm.with_structured_output(ResolvedEntityOutput)
    base_df = pipeline.df_entities.copy()

    index = faiss.IndexFlatL2(1024)
    index.add(np.array(base_df["description_embedded"].tolist()).astype(np.float32))

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(base_df["entity_name"].tolist())

    r_entities = pd.DataFrame()
    r_idx = []

    for _, entity in entity_df.iterrows():
        # semantic match for descriptions
        embed_description = await pipeline._embed_text(entity["entity_description"])
        _, idx = index.search(np.array([embed_description]).astype(np.float32), 20)
        desc_idx = idx.squeeze()

        # lexical match for names
        embed_name_sparse = tfidf.transform([entity["entity_name"]])
        name_similarities = cosine_similarity(embed_name_sparse, matrix).ravel()
        name_idx = np.argsort(name_similarities)[-10:][::-1]

        similar_idx = np.concatenate([name_idx, desc_idx])
        r_idx.extend(similar_idx)

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

        response = await pipeline._invoke_llm(output.ainvoke, messages)
        r_entities = pd.concat(
            [
                r_entities,
                pd.DataFrame([entity.model_dump() for entity in response.entities]),
            ]
        )

    base_df = base_df.drop(r_idx)
    base_df = pd.concat([base_df, r_entities], ignore_index=True)
    base_df = base_df.drop_duplicates(subset=["entity_name"], keep="last")

    return base_df


class RelationshipOutput(BaseModel):
    relationships: list[Relationship] = Field(
        title="Relationships",
        description="List of relationships identified.",
    )


async def _extract_relationships(
    pipeline: GraphPipeline, textunit_df: pd.DataFrame, entity_df: pd.DataFrame
) -> pd.DataFrame:
    output = pipeline.llm.with_structured_output(RelationshipOutput)

    entities_json = entity_df.to_json(orient="records")

    tasks = []
    for _, row in textunit_df.iterrows():
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
{row['text']}
                """,
            },
        ]
        tasks.append(pipeline._invoke_llm(output.ainvoke, messages))

    responses = await asyncio.gather(*tasks)

    relationship_df = pd.DataFrame(
        [
            relationship.model_dump()
            for response in responses
            for relationship in response.relationships
        ]
    )
    return relationship_df


class ClaimOutput(BaseModel):
    claims: list[Claim] = Field(
        title="Claims", description="List of claims identified in Step 2."
    )


async def _extract_claims(
    pipeline: GraphPipeline, textunit_df: pd.DataFrame, entity_df: pd.DataFrame
) -> pd.DataFrame:
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
{entity_df.to_json(orient="records")}

TEXT
----------
{row['text']}
        """,
            },
        ]
        tasks.append(pipeline._invoke_llm(output.ainvoke, messages))

    responses = await asyncio.gather(*tasks)

    claims_df = pd.DataFrame(
        [claim.model_dump() for response in responses for claim in response.claims]
    )

    return claims_df
