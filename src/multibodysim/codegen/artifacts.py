from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from .constants import GENERATED_EVALUATOR_ROOT, METADATA_FILENAME
from .metadata import (
    autowrap_eval_differentials_cache_key,
    make_json_serialisable,
)


def autowrap_eval_differentials_artifact_dir(
    dyn,
    cache_root: Path | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else GENERATED_EVALUATOR_ROOT
    return root / "autowrap_eval_differentials" / (
        autowrap_eval_differentials_cache_key(dyn)
    )


def metadata_path(artifact_dir: Path) -> Path:
    return artifact_dir / METADATA_FILENAME


def write_metadata(artifact_dir: Path, metadata: dict) -> None:
    metadata_path(artifact_dir).write_text(
        json.dumps(
            make_json_serialisable(metadata),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def read_metadata(artifact_dir: Path) -> dict | None:
    path = metadata_path(artifact_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compiled_module(module_name: str, module_file: Path) -> ModuleType:
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        existing_file = Path(getattr(existing_module, "__file__", ""))
        if existing_file.resolve() == module_file.resolve():
            return existing_module
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated module at {module_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_function_from_metadata(artifact_dir: Path, metadata: dict):
    module_file = artifact_dir / metadata["module_file"]
    if not module_file.exists():
        raise FileNotFoundError(
            f"Generated autowrap module does not exist: {module_file}",
        )

    module = compiled_module(metadata["module_name"], module_file)
    return getattr(module, metadata["function_name"])
