import json
import os

import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph


def load_parquet(file_name: str) -> pd.DataFrame:
    if not os.path.exists(file_name):
        return pd.DataFrame()
    return pd.read_parquet(file_name)


def load_json_graph(file_name: str) -> nx.Graph:
    if not os.path.exists(file_name):
        return nx.Graph()
    return json_graph.node_link_graph(json.load(open(file_name)))
