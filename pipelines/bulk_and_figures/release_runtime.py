"""External configuration boundary for archived CRC-TCR analysis modules.

No patient data or scientific sequence defaults are embedded in this module.
Set CRC_TCR_BULK_CONFIG to a JSON file outside the code repository.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _outside_checkout(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.resolve().is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"{label} must be outside the code repository")
    return path


@lru_cache(maxsize=8)
def _load_config(filename: str) -> dict:
    if not filename:
        raise RuntimeError("Set CRC_TCR_BULK_CONFIG to an external JSON configuration file")
    path = _outside_checkout(Path(filename), "Configuration")
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Configuration schema_version must be 1")
    literals_file = config.get("private_literals_file")
    if literals_file:
        source = Path(literals_file).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        source = _outside_checkout(source, "Private literals file")
        content = json.loads(source.read_text())
        literals = content.get("private_literals", content)
        config["private_literals"] = {**literals, **config.get("private_literals", {})}
    return config


def _value(section: str, key: str) -> str:
    config = _load_config(os.environ.get("CRC_TCR_BULK_CONFIG", ""))
    try:
        value = config[section][key]
    except KeyError as exc:
        raise KeyError(f"Missing external {section} field: {key}") from exc
    if not isinstance(value, str) or value.startswith("REPLACE_WITH_"):
        raise ValueError(f"External {section} field {key} must contain a configured string")
    return value


def release_path(key: str) -> str:
    value = _value("paths", key)
    if not Path(value).expanduser().is_absolute():
        raise ValueError(f"External path {key} must be absolute")
    return str(_outside_checkout(Path(value), "Input/output path"))


def private_literal(key: str) -> str:
    return _value("private_literals", key)
