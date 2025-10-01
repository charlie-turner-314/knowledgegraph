from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractionTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    confidence: Optional[float] = None
    source_reference: Optional[str] = None
    suggested_subject_label: Optional[str] = None
    suggested_object_label: Optional[str] = None


class ExtractionResponse(BaseModel):
    triples: list[ExtractionTriple]
    raw_response: dict
