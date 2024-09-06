import asyncio
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import cast

import networkx as nx
import pandas as pd
from graspologic.partition import hierarchical_leiden

from fargs.data import BaseGraphData
from fargs.models import CommunityReport

from .base import BaseFargs
from .base import FargsConfig

COMMUNITY_REPORT_MESSAGE = """
ENTITIES
----------
{entities_json}

RELATIONSHIPS
----------
{relationships_json}

CLAIMS
----------
{claims_json}
"""


class GraphTasks(BaseFargs, ABC):
    def __init__(self, config: FargsConfig):
        super().__init__(config)

    @property
    @abstractmethod
    def graph(self) -> BaseGraphData:
        pass

    async def build_graph(self):
        for _, row in self.entities:
            await self.graph.add_node(
                row["entity_name"],
                type=row["entity_type"],
                description=row["entity_description"],
            )

        for _, row in self.relationships:
            src_ent = self.entities.find_by_name("resolved", row["source_entity"])
            if not src_ent:
                aliases = self.entities.find_by_alias("resolved", row["source_entity"])
                if aliases:
                    src_ent = aliases[0]

            tgt_ent = self.entities.find_by_name("resolved", row["target_entity"])
            if not tgt_ent:
                aliases = self.entities.find_by_alias("resolved", row["target_entity"])
                if aliases:
                    tgt_ent = aliases[0]

            if src_ent is None or tgt_ent is None:
                continue

            await self.graph.add_edge(
                src_ent.entity_name,
                tgt_ent.entity_name,
                weight=row["relationship_strength"],
                description=row["relationship_description"],
            )
        return self.graph

    async def generate_community_reports(self, pipeline):
        graph = self.graph._data.copy()
        largest_components = sorted(
            nx.connected_components(graph), key=len, reverse=True
        )[:5]
        subgraphs = [graph.subgraph(c).copy() for c in largest_components]

        async def build_subgraph_communities(subgraph: nx.Graph) -> pd.DataFrame:
            subgraph = cast(nx.Graph, subgraph)
            subgraph_results = defaultdict(
                lambda: dict(
                    level=None,
                    title=None,
                    edges=set(),
                    nodes=set(),
                    sub_communities=[],
                )
            )
            subgraph_levels = defaultdict(set)

            community_mapping = hierarchical_leiden(
                subgraph,
                max_cluster_size=10,
                random_seed=0xDEADBEEF,
            )

            node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
            __levels = defaultdict(set)
            for partition in community_mapping:
                level_key = partition.level
                cluster_id = partition.cluster
                node_communities[partition.node].append(
                    {"level": level_key, "cluster": cluster_id}
                )
                __levels[level_key].add(cluster_id)

            node_communities = dict(node_communities)
            __levels = {k: len(v) for k, v in __levels.items()}

            for node_id, clusters in node_communities.items():
                subgraph.nodes[node_id]["clusters"] = clusters

            for node_id, node_data in subgraph.nodes(data=True):
                if "clusters" not in node_data:
                    continue

                clusters = node_data["clusters"]
                this_node_edges = subgraph.edges(node_id)

                for cluster in clusters:
                    level = cluster["level"]
                    cluster_key = str(cluster["cluster"])
                    subgraph_levels[level].add(cluster_key)

                    subgraph_results[cluster_key]["level"] = level
                    subgraph_results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    subgraph_results[cluster_key]["nodes"].add(node_id)
                    subgraph_results[cluster_key]["edges"].update(
                        [tuple(sorted(e)) for e in this_node_edges]
                    )

            ordered_levels = sorted(subgraph_levels.keys())
            for i, curr_level in enumerate(ordered_levels[:-1]):
                next_level = ordered_levels[i + 1]
                this_level_comms = subgraph_levels[curr_level]
                next_level_comms = subgraph_levels[next_level]
                # compute the sub-communities by nodes intersection
                for comm in this_level_comms:
                    subgraph_results[comm]["sub_communities"] = [
                        c
                        for c in next_level_comms
                        if subgraph_results[c]["nodes"].issubset(
                            subgraph_results[comm]["nodes"]
                        )
                    ]

            return subgraph_results

        async def build_community_report(community: dict):
            com_ents = [
                pipeline.entities.find_by_name("resolved", node)
                for node in community["nodes"]
            ]
            _com_rls = [
                pipeline.relationships.find_by_entities(edge[0], edge[1])
                for edge in community["edges"]
            ]
            com_rls = [r for sublist in _com_rls for r in sublist]

            _com_claims = [
                pipeline.claims.find_by_subject(node) for node in community["nodes"]
            ] + [pipeline.claims.find_by_object(node) for node in community["nodes"]]

            com_claims = [c for sublist in _com_claims for c in sublist]

            com_ents = [e for e in com_ents if e is not None]
            com_rls = [r for r in com_rls if r is not None]
            com_claims = [c for c in com_claims if c is not None]

            com_ents_json = [
                e.model_dump_json(
                    include={
                        "entity_name",
                        "entity_type",
                        "entity_description",
                    }
                )
                for e in com_ents
            ]
            com_rls_json = [
                r.model_dump_json(
                    include={
                        "source_entity",
                        "target_entity",
                        "relationship_description",
                        "relationship_strength",
                    }
                )
                for r in com_rls
            ]
            com_claims_json = [
                c.model_dump_json(
                    include={
                        "claim_subject",
                        "claim_object",
                        "claim_description",
                        "claim_type",
                        "claim_status",
                        "claim_period",
                    }
                )
                for c in com_claims
            ]
            messages = [
                {
                    "role": "system",
                    "content": COMMUNITY_REPORT_MESSAGE,
                },
                {
                    "role": "user",
                    "content": COMMUNITY_REPORT_MESSAGE.format(
                        entities_json="\n".join(com_ents_json),
                        relationships_json="\n".join(com_rls_json),
                        claims_json="\n".join(com_claims_json),
                    ),
                },
            ]
            output = self.llm.with_structured_output(CommunityReport)
            response = await self.invoke_llm(output.ainvoke, messages)
            return response

        results = []
        build_tasks = [
            asyncio.create_task(build_subgraph_communities(subg)) for subg in subgraphs
        ]
        for task in asyncio.as_completed(build_tasks):
            res = await task
            results.extend(res.values())

        for v in results:
            v["edges"] = list(v["edges"])
            v["nodes"] = list(v["nodes"])

        report_tasks = [
            asyncio.create_task(build_community_report(community))
            for community in results
        ]
        reports = await asyncio.gather(*report_tasks)
        for r in reports:
            print(r)
            print("\n\n")
        return reports


# def generate_community_description(pipeline, community: dict) -> str:
#     df_entities = [
#         pipeline.df_entities[pipeline.df_entities["entity_name"] == node]
#         for node in community["nodes"]
#     ]
#     df_relationships = [
#         pipeline.df_relationships[
#             (pipeline.df_relationships["source_entity"] == e1)
#             & (pipeline.df_relationships["target_entity"] == e2)
#         ]
#         for e1, e2 in community["edges"]
#     ]
#     entities = [ResolvedEntity(**e.to_dict()) for e in df_entities]


# def analyze_communities(
#     df: pd.DataFrame,
#     community_col: str = "community_id",
#     threshold: float = 0.5,
# ) -> list[str]:
#     """
#     Compresses communities and generates descriptions.

#     Args:
#         df: DataFrame with 'entity_id', 'community_id', and 'entity_name' columns.
#         community_col: Name of the community ID column.
#         threshold: Threshold for merging communities.

#     Returns:
#         A list of community descriptions.
#     """
#     df = build_community_hierarchy(df.copy(), community_col, threshold)
#     community_descriptions = []
#     for community_id in df[community_col].unique():
#         description = generate_community_description(df, community_id)
#         community_descriptions.append(f"Community {community_id}: {description}")
#     return community_descriptions
