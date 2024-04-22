from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class Format(Enum):
    JSON = "json"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


Image = List[str]


@dataclass
class Option:
    mirostat: int
    mirostat_eta: float
    mirostat_tau: float
    num_ctx: int
    repeat_last_n: int
    repeat_penalty: float
    temperature: float
    seed: int
    stop: str
    tfs_z: float
    num_predict: int
    top_k: int
    top_p: float


@dataclass
class Message:
    role: Role
    content: str
    image: Optional[Image]
