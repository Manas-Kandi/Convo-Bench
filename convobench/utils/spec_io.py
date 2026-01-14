"""Serialization utilities for benchmark spec primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def to_json_dict(model: BaseModel) -> dict[str, Any]:
    return json.loads(model.model_dump_json())


def save_json(model: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def load_json(model_cls: type[T], path: str | Path) -> T:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return model_cls.model_validate(data)
