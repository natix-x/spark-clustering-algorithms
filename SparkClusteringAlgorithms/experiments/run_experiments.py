#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import logging

import yaml

from yaml_utils.yaml_validator import validate_yaml_config_file
from spark_experiments.spark_experiments_generator import generate_experiments
from slurm.jobs_writer import write_configs, write_sbatch_files


def main() -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for Spark clustering experiments."
    )
    parser.add_argument("matrix", type=Path, help="Path to the YAML config file.")
    args = parser.parse_args()

    if not args.matrix.exists():
        raise FileNotFoundError(f"YAML config file not found: {args.matrix}")

    parsed_yaml_config_file = yaml.safe_load(args.matrix.read_text())
    validate_yaml_config_file(parsed_yaml_config_file)

    cfg_dir = os.path.expandvars(parsed_yaml_config_file["configs_dir"])
    log_dir = os.path.expandvars(parsed_yaml_config_file["log_dir"])
    out_dir = os.path.expandvars(parsed_yaml_config_file["output_dir"])

    for d in (cfg_dir, log_dir, out_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    repetitions = int(parsed_yaml_config_file.get("repetitions", 1))
    experiments = generate_experiments(parsed_yaml_config_file, repetitions)

    if repetitions > 1:
        logger.info(f"Repetitions: {repetitions}x -> {len(experiments)} runs in total")

    write_configs(experiments, cfg_dir, out_dir, parsed_yaml_config_file)

    sbatch_dir = str(Path(out_dir) / "sbatch")
    write_sbatch_files(experiments, cfg_dir, log_dir, out_dir, sbatch_dir, parsed_yaml_config_file)

    logger.info(f"Generated {len(experiments)} independent jobs.")
    logger.info(f"SBatch files:  {sbatch_dir}/")


if __name__ == "__main__":
    main()