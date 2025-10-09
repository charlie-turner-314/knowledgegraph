# Embedding persistence for NodeEmbeddingStore
# This module provides save/load utilities for the in-memory embedding index.
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

EMBEDDINGS_PATH = "data/embeddings.pkl"


def save_embeddings(
    labels: List[str],
    label_to_index: Dict[str, int],
    vectors: Optional[np.ndarray],
    embedding_dim: Optional[int],
    metadata: List[Dict[str, object]],
) -> None:
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(
            {
                "labels": labels,
                "label_to_index": label_to_index,
                "vectors": vectors,
                "embedding_dim": embedding_dim,
                "metadata": metadata,
            },
            f,
        )


def load_embeddings() -> Tuple[Optional[List[str]], Optional[Dict[str, int]], Optional[np.ndarray], Optional[int], Optional[List[Dict[str, object]]]]:
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        metadata = data.get("metadata")
        if metadata is None and data.get("labels"):
            # Backwards compatibility for previous format without metadata
            metadata = [
                {"label": label, "entity_type": None, "node_id": None}
                for label in data["labels"]
            ]
        return (
            data.get("labels"),
            data.get("label_to_index"),
            data.get("vectors"),
            data.get("embedding_dim"),
            metadata,
        )
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None, None, None, None, None
