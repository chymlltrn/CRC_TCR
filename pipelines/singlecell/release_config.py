"""Read local-only execution paths and source constants without embedded data.

Set CRC_TCR_CONFIG to a JSON file outside this repository. Empty examples are
schemas, not execution configurations. Loading does not read scientific data.
"""
from functools import lru_cache
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

@lru_cache(maxsize=1)
def load_config():
    name = os.environ.get('CRC_TCR_CONFIG', '')
    if not name:
        raise RuntimeError('Set CRC_TCR_CONFIG to your private JSON outside the repository; see config.example.json.')
    path = Path(name).expanduser().resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('Private execution configuration must remain outside the repository.')
    with path.open() as stream:
        config = json.load(stream)
    for section in ('paths', 'private_constants'):
        if not isinstance(config.get(section), dict):
            raise ValueError('Missing configuration object: '+section)
    return config

def release_path(key):
    value = load_config()['paths'].get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing private configuration path: '+key)
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError('Use an absolute path for configuration key: '+key)
    resolved = path.resolve()
    if resolved == REPOSITORY or REPOSITORY in resolved.parents:
        raise ValueError('Scientific input/output paths must remain outside the repository: '+key)
    return path

def private_value(key):
    value = load_config()['private_constants'].get(key)
    if value is None or value == '':
        raise ValueError('Missing private source constant: '+key)
    return value
