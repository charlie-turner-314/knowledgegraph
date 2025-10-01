from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .types import ChunkPayload, ParsedDocument
from .utils import compute_checksum


@dataclass
class ParserContext:
    path: Path
    media_type: str


class DocumentParser:
    media_types: List[str] = []
    extensions: List[str] = []

    def matches(self, ctx: ParserContext) -> bool:
        if self.media_types and ctx.media_type in self.media_types:
            return True
        if self.extensions and ctx.path.suffix.lower() in self.extensions:
            return True
        return False

    def parse(self, ctx: ParserContext) -> ParsedDocument:
        raise NotImplementedError


class TextParser(DocumentParser):
    extensions = [".txt", ".md"]
    media_types = ["text/plain", "text/markdown"]

    def parse(self, ctx: ParserContext) -> ParsedDocument:
        text = ctx.path.read_text(encoding="utf-8")
        chunks = [ChunkPayload(ordering=idx, text=line.strip()) for idx, line in enumerate(text.splitlines()) if line.strip()]
        if not chunks:
            chunks = [ChunkPayload(ordering=0, text=text)]
        return ParsedDocument(
            display_name=ctx.path.name,
            media_type=ctx.media_type,
            checksum=compute_checksum(ctx.path),
            chunks=chunks,
            original_path=ctx.path,
        )


class PDFParser(DocumentParser):
    extensions = [".pdf"]
    media_types = ["application/pdf"]

    def parse(self, ctx: ParserContext) -> ParsedDocument:
        import pdfplumber

        chunks: List[ChunkPayload] = []
        with pdfplumber.open(ctx.path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                chunks.append(
                    ChunkPayload(
                        ordering=idx,
                        text=text.strip(),
                        page_label=str(page.page_number),
                    )
                )
        return ParsedDocument(
            display_name=ctx.path.name,
            media_type=ctx.media_type,
            checksum=compute_checksum(ctx.path),
            chunks=chunks,
            original_path=ctx.path,
        )


class DocxParser(DocumentParser):
    extensions = [".docx"]
    media_types = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    def parse(self, ctx: ParserContext) -> ParsedDocument:
        import docx

        document = docx.Document(ctx.path)
        text_blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        chunks = [
            ChunkPayload(ordering=idx, text=block)
            for idx, block in enumerate(text_blocks)
        ]
        return ParsedDocument(
            display_name=ctx.path.name,
            media_type=ctx.media_type,
            checksum=compute_checksum(ctx.path),
            chunks=chunks,
            original_path=ctx.path,
        )


class PptxParser(DocumentParser):
    extensions = [".pptx"]
    media_types = [
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]

    def parse(self, ctx: ParserContext) -> ParsedDocument:
        from pptx import Presentation

        presentation = Presentation(ctx.path)
        chunks: List[ChunkPayload] = []
        for idx, slide in enumerate(presentation.slides):
            lines: List[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text = shape.text.strip()
                    if text:
                        lines.append(text)
            if not lines:
                continue
            chunks.append(
                ChunkPayload(
                    ordering=idx,
                    text="\n".join(lines),
                    page_label=f"slide-{idx + 1}",
                )
            )
        return ParsedDocument(
            display_name=ctx.path.name,
            media_type=ctx.media_type,
            checksum=compute_checksum(ctx.path),
            chunks=chunks,
            original_path=ctx.path,
        )


class ParserRegistry:
    def __init__(self, parsers: Optional[List[DocumentParser]] = None):
        self.parsers = parsers or [PDFParser(), DocxParser(), PptxParser(), TextParser()]

    def detect_media_type(self, path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.resolve().as_uri())
        return guessed or "application/octet-stream"

    def select_parser(self, path: Path) -> DocumentParser:
        media_type = self.detect_media_type(path)
        ctx = ParserContext(path=path, media_type=media_type)
        for parser in self.parsers:
            if parser.matches(ctx):
                return parser
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def parse(self, path: Path) -> ParsedDocument:
        media_type = self.detect_media_type(path)
        ctx = ParserContext(path=path, media_type=media_type)
        parser = self.select_parser(path)
        return parser.parse(ctx)


DEFAULT_REGISTRY = ParserRegistry()


def parse_document(path: Path) -> ParsedDocument:
    return DEFAULT_REGISTRY.parse(path)
