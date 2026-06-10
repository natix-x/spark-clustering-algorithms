from __future__ import annotations

import re

_REQUIRED_KEYS = ("name", "configs_dir", "log_dir", "output_dir", "jar_path", "matrix")
_REQUIRED_MATRIX_KEYS = ("nodes", "algorithm", "dataset")
_RESOURCE_KEYS = ("cpus_per_task", "executors_per_node", "mem")


def parse_mem_gb(mem_str: str) -> int:
    m = re.match(r'^(\d+)\s*[Gg]?$', str(mem_str).strip())
    if not m:
        raise ValueError(
            f"Wrong memory format: {mem_str!r}. Use e.g. '48G'."
        )
    return int(m.group(1))


def validate_yaml_config_file(parsed_yaml_config_file: dict) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in parsed_yaml_config_file]
    if missing:
        raise KeyError(f"Missing top-level keys in YAML: {missing}")

    matrix = parsed_yaml_config_file["matrix"]

    missing_matrix = [k for k in _REQUIRED_MATRIX_KEYS if k not in matrix]
    if missing_matrix:
        raise KeyError(
            f"Missing required keys in 'matrix': {missing_matrix}. "
            f"Required: {list(_REQUIRED_MATRIX_KEYS)}"
        )

    if "repetitions" in matrix:
        raise ValueError(
            "Repetitions should be top-level key in YAML, not in 'matrix'."
        )

    for key, values in matrix.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Matrix['{key}'] must be a non-empty list, "
                f"got: {type(values).__name__} = {values!r}"
            )

    repetitions = parsed_yaml_config_file.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")

    _validate_resources(parsed_yaml_config_file)


def _validate_resources(parsed_yaml_config_file: dict) -> None:
    """Resources can live in matrix.resources (per-experiment) or sbatch_defaults (fallback).

    At least one source must provide all of cpus_per_task, executors_per_node, mem.
    """
    matrix          = parsed_yaml_config_file["matrix"]
    sbatch_defaults = parsed_yaml_config_file.get("sbatch_defaults", {})

    if "resources" in matrix:
        for i, entry in enumerate(matrix["resources"]):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"matrix.resources[{i}] must be a dict, got {type(entry).__name__}"
                )
            missing = [k for k in _RESOURCE_KEYS if k not in entry]
            if missing:
                raise KeyError(
                    f"matrix.resources[{i}] missing keys: {missing}. "
                    f"Each entry must define: {list(_RESOURCE_KEYS)}"
                )
            _validate_resource_entry(entry, f"matrix.resources[{i}]")
        return

    missing = [k for k in _RESOURCE_KEYS if k not in sbatch_defaults]
    if missing:
        raise KeyError(
            f"Resources not in matrix.resources, so sbatch_defaults must define: "
            f"{missing}"
        )
    _validate_resource_entry(sbatch_defaults, "sbatch_defaults")


def _validate_resource_entry(entry: dict, source: str) -> None:
    cpus = entry["cpus_per_task"]
    if not isinstance(cpus, int) or cpus < 1:
        raise ValueError(f"{source}.cpus_per_task must be a positive integer, got {cpus!r}")

    execs = entry["executors_per_node"]
    if not isinstance(execs, int) or execs < 1:
        raise ValueError(f"{source}.executors_per_node must be a positive integer, got {execs!r}")

    parse_mem_gb(entry["mem"])
