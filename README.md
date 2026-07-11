# SparkClusteringAlgorithms
Master's thesis project - implementation and evaluation of Spark-based clustering algorithms.

## Table of contents:
* [General info](#general-info)
* [Description](#description)
* [Project structure](#project-structure)
* [Requirements](#requirements)
* [Experiments](#experiments)
* [Setup](#setup)

### General info
Part of master's thesis project: *"Performance and efficiency issues of the use of Big Data frameworks
for implementation of clustering algorithms".

The goal is to study the use of Big Data computing platforms for implementing clustering
algorithms, considering both **computational performance** — scalability, resource usage,
execution time — and the **effectiveness of the methods** in terms of the quality of the
results. Selected clustering algorithms and their Big Data implementations are compared
across algorithms as well as between Apache Spark and Apache Flink, in order to identify the main challenges,
outline possible solutions, and draw conclusions about the practical application of
clustering algorithms in Big Data environments.

### Description
The repository contains from-scratch Spark implementations of clustering algorithms from
three families, together with the Spark side of the benchmarking framework. It builds a
self-contained fat jar that runs **one config in, one result out**.

Experiment orchestration (matrix expansion, SLURM submission) and result analysis are
**engine-agnostic** and live in the shared
[`clustering-algorithms-benchmark`](https://github.com/natix-x/clustering-algorithms-benchmark)
repo, which drives both this Spark jar and the Flink jar via a common JSON contract
(`RunConfig` in, `RunResult` out). This repo only needs to keep producing results that
validate against that contract.

Implemented algorithms:
* **Centroid-based:** K-Means
* **Medoid-based:** PAM (naive), FastPAM, CLARA
* **Density-based:** DBSCAN (naive and grid-based variants)

Each algorithm supports pluggable distance metrics (Euclidean, Manhattan, Cosine).
Result quality is assessed with standard metrics such as the silhouette score, cluster
sizes, and noise fraction.

### Project structure
```
.
├── spark/                        # Scala/Spark SBT project (the fat jar)
│   ├── build.sbt                 # Scala/Spark build, assembly into a fat jar
│   ├── src/main/scala/clustering/
│   │   ├── core/                 # Clusterer / Trainer / Model abstractions
│   │   ├── algorithms/           # clustering algorithms implementations
│   │   ├── distance/             # Euclidean / Manhattan / Cosine metrics
│   │   ├── evaluation/           # clustering metrics
│   │   ├── utils/                # math, convergence, union-find, Spark helpers
│   │   └── benchmark/            # benchmark runner (--config), datasource, metrics, registry
│   └── local_run.sh              # spark-submit a single run locally
├── local_testing/                # JSON configs for local single-run testing
└── benchmark-results/            # local run outputs (one JSON result per run)
```

Experiment orchestration (`run_experiments.py`, YAML matrices, SLURM) and `analysis/`
now live in the [`clustering-algorithms-benchmark`](https://github.com/natix-x/clustering-algorithms-benchmark) repo.

### Requirements
* JDK 8 or 11
* Scala 2.12 / sbt (with `sbt-assembly`)
* Apache Spark 3.3.2 (Hadoop 3)

### Experiments
A run is described by a per-run JSON config (dataset, algorithm, evaluation, Spark conf)
that conforms to `contract/run_config.schema.json` in the benchmark repo. `BenchmarkRunner`
consumes exactly one config and produces exactly one JSON result file (conforming to
`contract/run_result.schema.json`), so failed runs are still recorded and array jobs never
silently lose data.

To run matrices on Ares (Cyfronet/PLGrid), use the benchmark repo:

```bash
# in clustering-algorithms-benchmark/
./slurm_run.sh spark experiment_configs/scaling_horizontal.yaml
```

The benchmark repo expands the YAML matrix into per-run configs + SLURM `sbatch` files
pointing at this repo's fat jar, and collects/analyses the results.

### Setup
Build the fat jar:
```bash
cd spark
sbt assembly        # -> target/scala-2.12/spark-clustering-benchmark.jar
```

Run a single benchmark locally (from repo root):
```bash
./spark/local_run.sh local_testing/experiment_configs/example.json
```
