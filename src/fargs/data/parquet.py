import asyncio
import os
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fargs.models import Claim
from fargs.models import Document
from fargs.models import Relationship
from fargs.models import ResolvedEntity
from fargs.models import ResolvedEntityOutput
from fargs.prompts import SIMILAR_ENTITY_RESOLUTION

from .base import BaseData
from .utils import load_parquet


@dataclass
class BaseParquetData(BaseData):
    def __post_init__(self):
        working_dir = self.config["working_dir"]
        self._data_file = os.path.join(working_dir, f"{self.namespace}.parquet")
        self._resolved_file = os.path.join(
            working_dir, f"{self.namespace}_resolved.parquet"
        )
        self._data: pd.DataFrame = load_parquet(self._data_file)
        self._resolved: pd.DataFrame = load_parquet(self._resolved_file)

    def __iter__(self):
        return self._resolved.iterrows()

    @property
    def is_empty(self) -> bool:
        return self._data.empty

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def resolve(self):
        raise NotImplementedError("This method must be implemented in a subclass.")

    async def insert(self):
        raise NotImplementedError("This method must be implemented in a subclass.")

    async def write(self, fmt: str = "parquet"):
        async with self._lock:
            if fmt == "parquet":
                try:
                    self._data.to_parquet(self._file_name, engine="auto")
                except Exception as e:
                    print(f"Error writing {self.namespace} to parquet: {e}")
            elif fmt == "csv":
                try:
                    self._data.to_csv(self._file_name + ".csv", index=False)
                except Exception as e:
                    print(f"Error writing {self.namespace} to csv: {e}")
            else:
                raise ValueError(f"Unsupported format: {fmt}")


@dataclass
class DocumentsParquetData(BaseParquetData):
    namespace: str = "documents"

    async def find_by_content_hash(self, content_hash: str) -> Document | None:
        df = self._data[self._data["content_hash"] == content_hash]
        if df.empty:
            return None
        return Document(**df.iloc[0].to_dict())

    async def insert(self, document: Document) -> bool:
        async with self._lock:
            if self.is_empty:
                self._data = pd.DataFrame([document.model_dump()])
                return True

            df = self._data[self._data["content_hash"] == document.content_hash]
            if df.empty:
                self._data = pd.concat(
                    [self._data, pd.DataFrame([document.model_dump()])],
                    ignore_index=True,
                ).drop_duplicates(subset=["doc_id"], keep="last")
                return True
            return False

    async def resolve(self):
        async with self._lock:
            if self.is_empty:
                return

            self._resolved = self._data.copy()


@dataclass
class TextUnitsParquetData(BaseParquetData):
    namespace: str = "text_units"

    async def insert(self, units: pd.DataFrame):
        async with self._lock:
            if self.is_empty:
                self._data = units
                return

            text_doc_ids = units["doc_id"].tolist()
            df = self._data[~self._data["doc_id"].isin(text_doc_ids)]
            self._data = pd.concat(
                [df, units],
                ignore_index=True,
            )

    async def resolve(self):
        async with self._lock:
            if self.is_empty:
                return

            self._resolved = self._data.copy()


@dataclass
class EntitiesParquetData(BaseParquetData):
    namespace: str = "entities"

    def find_by_name(self, file: str, name: str) -> ResolvedEntity | None:
        if file == "data":
            name = name.upper()
            df = self._data[self._data["entity_name"] == name]
            if df.empty:
                return None
            return ResolvedEntity(**df.iloc[0].to_dict())
        if file == "resolved":
            name = name.upper()
            df = self._resolved[self._resolved["entity_name"] == name]
            if df.empty:
                return None
            return ResolvedEntity(**df.iloc[0].to_dict())

    def find_by_alias(self, file: str, alias: str) -> list[ResolvedEntity] | None:
        if file == "data":
            alias = alias.upper()
            df = self._data[
                self._data["aliases"].apply(lambda aliases: alias in aliases)
            ]
            if df.empty:
                return None
            return [ResolvedEntity(**row.to_dict()) for _, row in df.iterrows()]
        if file == "resolved":
            alias = alias.upper()
            df = self._resolved[
                self._resolved["aliases"].apply(lambda aliases: alias in aliases)
            ]
            if df.empty:
                return None
            return [ResolvedEntity(**row.to_dict()) for _, row in df.iterrows()]

    async def insert(self, entities: pd.DataFrame):
        async with self._lock:
            if self.is_empty:
                self._data = entities
                return

            self._data = pd.concat([self._data, entities], ignore_index=True)

    async def resolve(self, pipeline):
        async with self._lock:
            df = self._data.copy()

            def get_combined_similarity(idx, top_k=10):
                tfidf_sim = cosine_similarity(
                    tfidf_matrix[idx : idx + 1], tfidf_matrix
                ).flatten()

                _, faiss_indices = index.search(
                    description_embeddings[idx : idx + 1], top_k
                )
                faiss_sim = np.zeros(len(df))
                faiss_sim[faiss_indices[0]] = 1 - np.arange(top_k) / top_k

                combined_sim = 0.4 * tfidf_sim + 0.6 * faiss_sim
                return combined_sim

            async def resolve_cluster(cluster: pd.Series):
                cluster_rows = df[df["e_id"].isin(cluster["e_id"])]
                if len(cluster_rows) == 1:
                    return cluster_rows.to_dict(orient="records")

                cluster_entities = [
                    ResolvedEntity(**row.to_dict())
                    for _, row in cluster_rows.iterrows()
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
                                    include={
                                        "entity_name",
                                        "entity_type",
                                        "entity_description",
                                    }
                                )
                                for entity in cluster_entities
                            ]
                        ),
                    },
                ]
                output = pipeline.llm.with_structured_output(ResolvedEntityOutput)
                response = await pipeline.invoke_llm(output.ainvoke, messages)

                embedded_entities = [
                    asyncio.create_task(pipeline.embed_entity(e))
                    for e in response.entities
                ]
                embedded_entities = await asyncio.gather(*embedded_entities)
                return embedded_entities

            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(df["entity_name"].tolist())

            description_embeddings = np.array(
                df["description_embedded"].tolist()
            ).astype(np.float32)
            faiss.normalize_L2(description_embeddings)
            index = faiss.IndexFlatIP(description_embeddings.shape[1])
            index.add(description_embeddings)

            similarity_matrix = np.array(
                [get_combined_similarity(i) for i in range(len(df))]
            )
            distance_matrix = 1 - similarity_matrix

            np.fill_diagonal(distance_matrix, 0)  # Ensure self-distance is 0

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,
                metric="precomputed",
                linkage="average",
            )
            clusters = clustering.fit_predict(distance_matrix)
            df["cluster"] = clusters

            merged_df = df.groupby("cluster").agg({"e_id": list})

            resolve_clusters = [
                asyncio.create_task(resolve_cluster(cluster))
                for _, cluster in merged_df.iterrows()
            ]
            entities = []
            for task in asyncio.as_completed(resolve_clusters):
                es = await task
                entities.extend(es)

            self._resolved = pd.DataFrame(entities)


@dataclass
class RelationshipsParquetData(BaseParquetData):
    namespace: str = "relationships"

    def find_by_entities(
        self, source_entity: str, target_entity: str
    ) -> list[Relationship]:
        if self.is_empty:
            return []
        source_entity = source_entity.upper()
        target_entity = target_entity.upper()
        df = self._data[
            (self._data["source_entity"] == source_entity)
            & (self._data["target_entity"] == target_entity)
        ]
        return [Relationship(**row.to_dict()) for _, row in df.iterrows()]

    def find_by_entity_ids(self, source_id: str, target_id: str) -> pd.DataFrame | None:
        if self.is_empty:
            return None
        df = self._data[
            (self._data["source_id"] == source_id)
            & (self._data["target_id"] == target_id)
        ]
        return df if not df.empty else None

    async def insert(self, pipeline, relationships: pd.DataFrame):
        async def insert_one(row: pd.Series):
            async with self._lock:
                if self.is_empty:
                    self._data = pd.DataFrame([row.to_dict()])
                    return

                existing_relationship = self.find_by_entity_ids(
                    row["source_id"], row["target_id"]
                )
                if existing_relationship:
                    row["relationship_description"] = (
                        " ".join(
                            existing_relationship["relationship_description"].tolist()
                        )
                        + " "
                        + row["relationship_description"]
                    )
                    strengths = existing_relationship["relationship_strength"].tolist()
                    strengths.append(row["relationship_strength"])
                    row["relationship_strength"] = np.mean(strengths)
                    row["r_source_units"] = list(
                        set(
                            existing_relationship["r_source_units"].explode().tolist()
                            + row["r_source_units"].explode().tolist()
                        )
                    )
                    row_dict = await pipeline.embed_relationship(
                        Relationship(**row.to_dict())
                    )
                    row_df = pd.DataFrame([row_dict])
                    self._data = pd.concat([self._data, row_df], ignore_index=True)
                    self._data = self._data.drop_duplicates(
                        subset=["source_id", "target_id"], keep="last"
                    )

                else:
                    self._data = pd.concat(
                        [self._data, pd.DataFrame([row.to_dict()])], ignore_index=True
                    )

        await asyncio.gather(*[insert_one(row) for _, row in relationships.iterrows()])

    async def resolve(self):
        async with self._lock:
            if self.is_empty:
                return

            self._resolved = self._data.copy()


@dataclass
class ClaimsParquetData(BaseParquetData):
    namespace: str = "claims"

    def find_by_subject(self, entity_subject: str) -> list[Claim]:
        """
        Find claims by subject.

        Args:
            subject: The subject of the claims to find.

        Returns:
            A list of claims if found, otherwise an empty list.
        """
        entity_subject = entity_subject.upper()
        df = self._data[self._data["claim_subject"] == entity_subject]
        if df.empty:
            return []
        return [Claim(**row.to_dict()) for _, row in df.iterrows()]

    def find_by_object(self, entity_object: str) -> list[Claim]:
        """
        Find claims by object.

        Args:
            object: The object of the claims to find.

        Returns:
            A list of claims if found, otherwise an empty list.
        """
        entity_object = entity_object.upper()
        df = self._data[self._data["claim_object"] == entity_object]
        if df.empty:
            return []
        return [Claim(**row.to_dict()) for _, row in df.iterrows()]

    async def insert(self, claims: pd.DataFrame):
        async with self._lock:
            self._data = pd.concat([self._data, claims], ignore_index=True)

    async def resolve(self):
        async with self._lock:
            if self.is_empty:
                return

            self._resolved = self._data.copy()
