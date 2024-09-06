import json
import os
from dataclasses import dataclass
from typing import Any

import aiofiles
import networkx as nx

from .base import BaseData
from .utils import load_json_graph


@dataclass
class BaseGraphData(BaseData):
    namespace: str = "graph"

    def __post_init__(self):
        working_dir = self.config["working_dir"]
        self._file_name = os.path.join(working_dir, f"{self.namespace}.json")
        self._data: nx.Graph = load_json_graph(self._file_name)

    async def add_node(self, node: Any, **kwargs):
        async with self._lock:
            self._data.add_node(node, **kwargs)

    async def add_edge(self, u: Any, v: Any, **kwargs):
        async with self._lock:
            self._data.add_edge(u, v, **kwargs)

    async def write(self):
        async with self._lock:
            async with aiofiles.open(self._file_name, "w") as f:
                await f.write(
                    json.dumps(nx.readwrite.json_graph.node_link_data(self._data))
                )
