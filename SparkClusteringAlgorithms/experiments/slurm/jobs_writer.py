from __future__ import annotations

import json
import os
import logging
from collections import defaultdict
from pathlib import Path

from slurm.sbatch_script_template import SBATCH_TEMPLATE
from spark_experiments.spark_experiments_generator import Experiment
from spark_experiments.spark_resources_resolver import SparkResources, compute_resources
from yaml_utils.yaml_validator import parse_mem_gb


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _resolve_resources(run: Experiment) -> SparkResources:
    res = run.cell["resources"]
    return compute_resources(
        total_mem_gb=parse_mem_gb(res["mem"]),
        executors_per_node=int(res["executors_per_node"]),
    )


def write_configs(
        runs: list[Experiment],
        cfg_dir: str,
        out_dir: str,
        experiments: dict,
) -> None:
    spark_conf      = dict(experiments.get("spark_conf_defaults", {}))
    evaluation      = dict(experiments.get("evaluation_defaults", {}))
    sbatch_defaults = experiments.get("sbatch_defaults", {})

    for run in runs:
        res          = run.cell["resources"]
        spark_res    = _resolve_resources(run)

        experiment_metadata = {
            "nodes":              str(run.nodes),
            "cpus_per_task":      str(res["cpus_per_task"]),
            "executors_per_node": str(res["executors_per_node"]),
            "total_executors":    str(run.nodes * int(res["executors_per_node"])),
            "mem":                str(res["mem"]),
            "walltime":           str(sbatch_defaults.get("walltime", "")),
            "partition":          str(sbatch_defaults.get("partition", "")),
            "spark_module":       str(sbatch_defaults.get("spark_module", "")),
            "worker_mem_gb":      str(spark_res.worker_pool_gb),
            "executor_mem_gb":    str(spark_res.executor_gb),
            "driver_mem_gb":      str(spark_res.driver_gb),
            "master_mem_gb":      str(spark_res.master_gb),
        }

        config = {
            "runId":              run.run_id,
            "profile":            "ares",
            "outputDir":          out_dir,
            "dataset":            dict(run.cell["dataset"]),
            "algorithm":          dict(run.cell["algorithm"]),
            "evaluation":         evaluation,
            "sparkConf":          spark_conf,
            "experimentMetadata": experiment_metadata,
        }
        path = Path(cfg_dir) / f"{run.run_id}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def write_sbatch_files(
        runs: list[Experiment],
        cfg_dir: str,
        log_dir: str,
        out_dir: str,
        sbatch_dir: str,
        doc: dict,
) -> None:
    """
    Generates SLURM job scripts for each run and writes them to the specified directory.
    """
    sbatch_path = Path(sbatch_dir)
    sbatch_path.mkdir(parents=True, exist_ok=True)

    sbatch_defaults = doc.get("sbatch_defaults", {})
    non_resource_defaults = {
        k: v for k, v in sbatch_defaults.items()
        if k not in ("executors_per_node", "cpus_per_task", "mem")
    }
    jar = os.path.expandvars(doc["jar_path"])

    submit_lines = ["#!/bin/bash", "set -e", ""]

    groups: dict[int, list[Experiment]] = defaultdict(list)
    for run in runs:
        groups[run.nodes].append(run)

    _log_resource_summaries(runs)

    for nodes in sorted(groups):
        for run in groups[nodes]:
            res         = run.cell["resources"]
            spark_res   = _resolve_resources(run)
            config_path = Path(cfg_dir) / f"{run.run_id}.json"
            script_path = sbatch_path / f"{run.run_id}.sbatch"

            script_path.write_text(SBATCH_TEMPLATE.format(
                name=doc["name"],
                nodes=nodes,
                run_id=run.run_id,
                config_path=str(config_path),
                log_dir=log_dir,
                output_dir=out_dir,
                jar=jar,
                cpus_per_task=int(res["cpus_per_task"]),
                mem=res["mem"],
                executors_per_node=int(res["executors_per_node"]),
                worker_mem=spark_res.worker_pool_gb,
                driver_mem=spark_res.driver_gb,
                executor_mem=spark_res.executor_gb,
                **non_resource_defaults,
            ))
            script_path.chmod(0o755)
            submit_lines.append(f"sbatch {script_path}")

        logger.info(f"nodes={nodes}: {len(groups[nodes])} sbatch files generated.")

    submit_all_file = sbatch_path / "submit_all.sh"
    submit_all_file.write_text("\n".join(submit_lines) + "\n")
    submit_all_file.chmod(0o755)


def _log_resource_summaries(runs: list[Experiment]) -> None:
    seen: set[tuple] = set()
    for run in runs:
        res = run.cell["resources"]
        key = (res["cpus_per_task"], res["executors_per_node"], res["mem"])
        if key in seen:
            continue
        seen.add(key)
        logger.info(
            f"Resource profile: cpus_per_task={key[0]}, "
            f"executors_per_node={key[1]}, mem={key[2]}"
        )
        _resolve_resources(run).print_summary()