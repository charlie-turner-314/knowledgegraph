from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.data import models


@dataclass
class QueryMatch:
    subject: str
    predicate: str
    object: str
    source_document: str | None
    page_label: str | None


@dataclass
class QueryResult:
    answers: List[str]
    matches: List[QueryMatch]
    note: str | None = None


class QueryService:
    def __init__(self, session: Session):
        self.session = session

    def ask(self, question: str) -> QueryResult:
        keywords = {word.lower() for word in re.findall(r"[A-Za-z0-9]+", question) if len(word) > 3}
        if not keywords:
            return QueryResult(
                answers=[],
                matches=[],
                note="Enter a more descriptive question to search the knowledge graph.",
            )

        stmt = (
            select(models.Edge)
            .options(
                selectinload(models.Edge.subject),
                selectinload(models.Edge.object),
                selectinload(models.Edge.sources)
                .selectinload(models.EdgeSource.document_chunk)
                .selectinload(models.DocumentChunk.document),
            )
        )
        edges = self.session.exec(stmt).all()

        matches: List[QueryMatch] = []
        for edge in edges:
            haystack = " ".join([
                edge.subject.label if edge.subject else "",
                edge.predicate,
                edge.object.label if edge.object else "",
            ]).lower()
            if keywords.intersection(haystack.split()):
                source = edge.sources[0] if edge.sources else None
                document = None
                page = None
                if source and source.document_chunk:
                    if source.document_chunk.document:
                        document = source.document_chunk.document.display_name
                    page = source.document_chunk.page_label
                matches.append(
                    QueryMatch(
                        subject=edge.subject.label if edge.subject else "",
                        predicate=edge.predicate,
                        object=edge.object.label if edge.object else "",
                        source_document=document,
                        page_label=page,
                    )
                )

        answers = [f"Found {len(matches)} triples matching keywords {sorted(keywords)}"] if matches else []
        return QueryResult(
            answers=answers,
            matches=matches,
            note="LLM-backed reasoning not yet connected; using keyword match as fallback.",
        )
