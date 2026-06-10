# TODO: refactor later on
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Experiment:
    run_id: str
    nodes: int
    cell: dict[str, Any]
    rep: int


def generate_experiments(parsed_yaml_config_file: dict, repetitions: int = 1) -> list[Experiment]:
    matrix = dict(parsed_yaml_config_file["matrix"])
    if "resources" not in matrix:
        defaults = parsed_yaml_config_file.get("sbatch_defaults", {})
        matrix["resources"] = [{
            "cpus_per_task":      defaults["cpus_per_task"],
            "executors_per_node": defaults["executors_per_node"],
            "mem":                defaults["mem"],
        }]

    keys   = list(matrix)
    name   = parsed_yaml_config_file["name"]
    experiments:  list[Experiment] = []
    counters: dict[int, int] = {}

    for combo in itertools.product(*(matrix[k] for k in keys)):
        cell = {
            k: (os.path.expandvars(v) if isinstance(v, str) else v)
            for k, v in zip(keys, combo)
        }
        nodes     = cell["nodes"]
        base_hash = _stable_hash(cell)

        for rep in range(repetitions):
            h = _rep_hash(cell, rep) if repetitions > 1 else base_hash

            idx            = counters.get(nodes, 0)
            counters[nodes] = idx + 1
            run_id         = f"{name}-nodes{nodes}-{idx:04d}-{h}"

            experiments.append(Experiment(run_id=run_id, nodes=nodes, cell=cell, rep=rep))

    return experiments

def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(
        f"Unsupported type in matrix: {type(obj).__name__}. "
        f"Only dicts, lists, tuples, and primitives are allowed."
    )


def _stable_hash(cell: dict) -> str:
    canonical = json.dumps(_normalize(cell), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]


def _rep_hash(cell: dict, rep: int) -> str:
    payload = json.dumps(
        {"cell": _normalize(cell), "rep": rep},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:8]
