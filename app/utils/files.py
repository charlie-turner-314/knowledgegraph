from __future__ import annotations

import secrets
from pathlib import Path
from typing import BinaryIO

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def store_uploaded_file(filename: str, file_obj: BinaryIO) -> Path:
    """Persist an uploaded file to disk and return the stored path."""
    token = secrets.token_hex(4)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    target = UPLOAD_DIR / f"{token}_{safe_name}"
    file_obj.seek(0)
    with target.open("wb") as handle:
        handle.write(file_obj.read())
    return target
