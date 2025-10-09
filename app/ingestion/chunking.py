from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import tiktoken

from .types import ChunkPayload


DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 120
DEFAULT_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class TextBlock:
	"""Unit of source text to be chunked."""

	text: str
	page_label: Optional[str] = None


def _get_encoder(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
	try:
		return tiktoken.get_encoding(encoding_name)
	except Exception:  # noqa: BLE001
		return tiktoken.get_encoding("cl100k_base")


def _split_into_sentences(text: str) -> List[str]:
	if not text.strip():
		return []
	normalized = re.sub(r"\s+", " ", text.strip())
	pieces = re.split(r"(?<=[.!?])(\s+(?=[A-Z0-9]))", normalized)
	sentences: List[str] = []
	buffer = []
	for piece in pieces:
		if not piece:
			continue
		buffer.append(piece)
		if re.search(r"[.!?]\s*$", piece):
			sentence = "".join(buffer).strip()
			if sentence:
				sentences.append(sentence)
			buffer = []
	remainder = "".join(buffer).strip()
	if remainder:
		sentences.append(remainder)
	return sentences or [normalized]


def _aggregate_page_label(labels: Iterable[Optional[str]]) -> Optional[str]:
	filtered = [label for label in {label for label in labels if label} if label]
	if not filtered:
		return None
	if len(filtered) == 1:
		return filtered[0]
	sorted_labels = sorted(filtered)
	return ", ".join(sorted_labels[:4]) + ("..." if len(sorted_labels) > 4 else "")


def chunk_blocks(
	blocks: Sequence[TextBlock],
	*,
	max_tokens: int = DEFAULT_MAX_TOKENS,
	overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
	encoding_name: str = DEFAULT_ENCODING,
) -> List[ChunkPayload]:
	if max_tokens <= 0:
		raise ValueError("max_tokens must be positive")
	overlap_tokens = max(0, min(overlap_tokens, math.floor(max_tokens * 0.9)))
	encoder = _get_encoder(encoding_name)
	chunks: List[ChunkPayload] = []

	current_sentences: List[tuple[str, int, Optional[str]]] = []
	current_tokens = 0

	def flush_chunk() -> None:
		nonlocal current_sentences, current_tokens
		if not current_sentences:
			return
		text_parts = [sentence for sentence, _, _ in current_sentences]
		chunk_text = " ".join(text_parts).strip()
		if not chunk_text:
			current_sentences = []
			current_tokens = 0
			return
		labels = [label for _, _, label in current_sentences]
		chunk_payload = ChunkPayload(
			ordering=len(chunks),
			text=chunk_text,
			page_label=_aggregate_page_label(labels),
			token_count=current_tokens,
		)
		chunks.append(chunk_payload)

		if overlap_tokens <= 0 or current_tokens <= overlap_tokens:
			current_sentences = []
			current_tokens = 0
			return

		overlap_sentences: List[tuple[str, int, Optional[str]]] = []
		running = 0
		for sentence, token_len, label in reversed(current_sentences):
			overlap_sentences.insert(0, (sentence, token_len, label))
			running += token_len
			if running >= overlap_tokens:
				break
		current_sentences = overlap_sentences
		current_tokens = sum(token_len for _, token_len, _ in current_sentences)

	for block in blocks:
		if not block.text or not block.text.strip():
			continue
		sentences = _split_into_sentences(block.text)
		for sentence in sentences:
			token_ids = encoder.encode(sentence)
			token_len = len(token_ids)
			if token_len == 0:
				continue
			if token_len > max_tokens:
				start = 0
				while start < token_len:
					end = min(start + max_tokens, token_len)
					fragment_tokens = token_ids[start:end]
					fragment_text = encoder.decode(fragment_tokens).strip()
					if fragment_text:
						current_sentences.append((fragment_text, len(fragment_tokens), block.page_label))
						current_tokens += len(fragment_tokens)
						flush_chunk()
					start = end - overlap_tokens if overlap_tokens and end < token_len else end
				continue
			if current_tokens + token_len > max_tokens and current_sentences:
				flush_chunk()
			current_sentences.append((sentence, token_len, block.page_label))
			current_tokens += token_len

	flush_chunk()
	return chunks


def chunk_text(
	text: str,
	*,
	page_label: Optional[str] = None,
	max_tokens: int = DEFAULT_MAX_TOKENS,
	overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
	encoding_name: str = DEFAULT_ENCODING,
) -> List[ChunkPayload]:
	block = TextBlock(text=text, page_label=page_label)
	return chunk_blocks([block], max_tokens=max_tokens, overlap_tokens=overlap_tokens, encoding_name=encoding_name)
