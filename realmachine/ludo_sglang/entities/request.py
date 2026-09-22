"""Request/response dataclasses for the Ludo-SGLang framework."""
from dataclasses import dataclass, field


@dataclass
class LudoRequest:
    prompt: str
    max_new_tokens: int = 128
    request_id: str = ""


@dataclass
class LudoResponse:
    request_id: str
    output_text: str
    dp_rank: int
    ludo_hit: bool
    cached_tokens: int
    prompt_tokens: int
