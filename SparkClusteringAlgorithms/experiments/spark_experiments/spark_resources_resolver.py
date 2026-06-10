# TODO: document this code better, check once more time memory calculations
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparkResources:
    total_gb:           int
    os_margin_gb:       int
    master_gb:          int
    driver_gb:          int
    worker_pool_gb:     int
    executor_gb:        int
    executors_per_node: int

    def print_summary(self) -> None:
        logger.info(
            f"--- Spark resources: --mem={self.total_gb}G, "
            f"executors_per_node={self.executors_per_node} ---"
        )
        logger.info(f"  OS margin:          {self.os_margin_gb} GB")
        logger.info(f"  Spark Master:       {self.master_gb} GB")
        logger.info(f"  Spark Driver:       {self.driver_gb} GB")
        logger.info(f"  Worker pool:        {self.worker_pool_gb} GB")
        logger.info(
            f"  Memory/executor:    {self.executor_gb} GB "
            f"({self.executors_per_node}/node, JVM +10%)"
        )
        used = (
                self.os_margin_gb
                + self.master_gb
                + self.driver_gb
                + self.worker_pool_gb
        )
        logger.info(f"  Total used:         {used}/{self.total_gb} GB")


def compute_resources(total_mem_gb: int, executors_per_node: int) -> SparkResources:
    """
    Computes memory allocation for each Spark component on a node.

    Breakdown:
        total = os_margin + master_overhead + driver + worker_pool
        executor_mem = (worker_pool / executors_per_node) / 1.10
    """
    master_gb = 1
    os_gb     = max(2, int(total_mem_gb * 0.06))
    driver_gb = max(2, int(total_mem_gb * 0.10))
    remaining = total_mem_gb - master_gb - os_gb - driver_gb

    if remaining <= 0:
        raise ValueError(
            f"Too little memory ({total_mem_gb}G) after deducting OS margin, "
            f"Spark Master, and Driver. "
            "Increase SBATCH --mem."
        )

    worker_pool_gb = int(remaining * 0.95)
    executor_gb    = max(1, int((worker_pool_gb / executors_per_node) / 1.10))

    return SparkResources(
        total_gb=total_mem_gb,
        os_margin_gb=os_gb,
        master_gb=master_gb,
        driver_gb=driver_gb,
        worker_pool_gb=worker_pool_gb,
        executor_gb=executor_gb,
        executors_per_node=executors_per_node,
    )
