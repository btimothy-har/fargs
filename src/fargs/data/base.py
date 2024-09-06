import asyncio
from dataclasses import dataclass


@dataclass
class BaseData:
    config: dict
    namespace: str
    _lock: asyncio.Lock = asyncio.Lock()
