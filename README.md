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
three families, together with a benchmarking framework used to run reproducible
experiments and collect performance and quality metrics.

Implemented algorithms:
* **Centroid-based:** K-Means
* **Medoid-based:** PAM (naive), FastPAM, CLARA
* **Density-based:** DBSCAN (naive and grid-based variants)

Each algorithm supports pluggable distance metrics (Euclidean, Manhattan, Cosine).
Result quality is assessed with standard metrics such as the silhouette score, cluster
sizes, and noise fraction.

### Project structure
```
SparkClusteringAlgorithms/
├── build.sbt                     # Scala/Spark build, assembly into a fat jar
├── src/main/scala/clustering/
│   ├── core/                     # Clusterer / Trainer / Model abstractions
│   ├── algorithms/               # clustering algorithms implementations
│   ├── distance/                 # Euclidean / Manhattan / Cosine metrics
│   ├── evaluation/               # clustering metrics
│   ├── utils/                    # math, convergence, union-find, Spark helpers
│   └── benchmark/                # benchmark runner, datasource, metrics, registry
├── experiments/                  # Python: generate & submit SLURM job arrays
│   ├── run_experiments.py        # entry point (YAML matrix -> per-run configs + sbatch)
│   └── experiment_configs/       # YAML experiment matrices (scaling, comparison, ...)
├── local_testing/                # JSON configs for local single-run testing
└── local_run.sh                  # spark-submit a single run locally
analysis/                         # Python analysis & figures from collected results
```

### Requirements
* JDK 8 or 11
* Scala 2.12 / sbt (with `sbt-assembly`)
* Apache Spark 3.3.2 (Hadoop 3)
* Python 3.12+ (`pyyaml`; analysis: `numpy`, `pandas`, `matplotlib`, `scikit-learn`)

### Experiments
A run is described by a per-run JSON config (dataset, algorithm, evaluation, Spark conf).
`BenchmarkRunner` consumes exactly one config and produces exactly one JSON result file,
so failed runs are still recorded and array jobs never silently lose data.

For cluster experiments on Ares (Cyfronet/PLGrid), an experiment matrix is defined in YAML
(varying nodes, resources, algorithms, datasets, with repetitions). `run_experiments.py`
expands the matrix into independent per-run configs and SLURM `sbatch` files:

```bash
cd SparkClusteringAlgorithms/experiments
python run_experiments.py experiment_configs/scaling_horizontal.yaml
```

Provided matrices cover horizontal scaling, vertical scaling (CPU / memory), algorithm
comparison, and cluster topology. Collected JSON results are then analysed under
`analysis/` to produce the scaling, efficiency and reproducibility figures.

### Setup
Build the fat jar:
```bash
cd SparkClusteringAlgorithms
sbt assembly        # -> target/scala-2.12/spark-clustering-benchmark.jar
```

Run a single benchmark locally:
```bash
./local_run.sh local_testing/experiment_configs/example.json
```
