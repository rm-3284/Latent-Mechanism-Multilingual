"""Compatibility shim. The Antonyms dataset lives at datasets/antonyms.py."""
from __future__ import annotations

import importlib.util

from lib.paths import antonyms_path

_spec = importlib.util.spec_from_file_location("antonyms_dataset", antonyms_path())
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load Antonyms dataset from {antonyms_path()}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

small_data = _mod.small_data
big_data = _mod.big_data
train_data = _mod.train_data
test_data = _mod.test_data
