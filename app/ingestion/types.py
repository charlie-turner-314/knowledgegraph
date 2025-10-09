from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ChunkPayload:
    ordering: int
    text: str
    page_label: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None


@dataclass
class ParsedDocument:
    display_name: str
    media_type: str
    checksum: str
    chunks: List[ChunkPayload]
    original_path: Optional[Path] = None
    extra: Optional[dict] = None

    def iter_chunks(self) -> Iterable[ChunkPayload]:
        return iter(self.chunks)
