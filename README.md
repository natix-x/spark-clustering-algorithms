# spark-clustering-algorithms

Part of the Master thesis 'Performance and efficiency issues of the use of Big Data frameworks for implementation of clustering algorithms'.

## Table of contents
* [General info](#general-info)
* [Architecture](#architecture)
* [Project structure](#project-structure)
* [Requirements](#requirements)
* [Usage](#usage)
* [Contract](#contract)


## General info

From-scratch **Scala / Spark** implementations of clustering algorithms, plus the
Spark side of the benchmarking framework. Builds a self-contained fat jar that runs
**one config in, one result out**.

This repo owns only the algorithms and the Spark job. Experiment orchestration (matrix
expansion, SLURM submission), the exchange **contract**, and result analysis are
engine-agnostic and live in the shared repo: [`clustering-algorithms-benchmark`](https://github.com/natix-x/clustering-algorithms-benchmark).

> **Note:** Some documentation and diagrams are in Polish, as the master thesis they accompany is written in Polish.

## Implemented algorithms

Registry name (`config.algorithm.name`) → factory in
[`AlgorithmRegistry.scala`](spark/src/main/scala/clustering/benchmark/registry/AlgorithmRegistry.scala):

- **`kmeans`** — Lloyd's k-means, `geometry: euclidean|spherical`, `refine: none|breathing`.
  [`KMeans.scala`](spark/src/main/scala/clustering/algorithms/kmeans/KMeans.scala) ·
  [`LloydKMeans.scala`](spark/src/main/scala/clustering/algorithms/kmeans/LloydKMeans.scala) ·
  [`BreathingKMeans.scala`](spark/src/main/scala/clustering/algorithms/kmeans/BreathingKMeans.scala)
- **`bisectingkmeans`** — divisive 2-means tree.
  [`BisectingKMeans.scala`](spark/src/main/scala/clustering/algorithms/kmeans/hierarchical/BisectingKMeans.scala)
- **`distfastpam`** — distributed FastPAM1 k-medoids (broadcast + `treeAggregate`).
  [`DistributedFastPAM.scala`](spark/src/main/scala/clustering/algorithms/kmedoids/distributed/DistributedFastPAM.scala)
- **`clara`** — PAM over random samples, `inner: fastpam|fasterpam`.
  [`CLARA.scala`](spark/src/main/scala/clustering/algorithms/kmedoids/hybrid/CLARA.scala) ·
  [`FastPAM.scala`](spark/src/main/scala/clustering/algorithms/kmedoids/local/FastPAM.scala) ·
  [`FasterPAM.scala`](spark/src/main/scala/clustering/algorithms/kmedoids/local/FasterPAM.scala)
- **`pamae`** — CLARA seeding + distributed Voronoi refinement (PAMAE, KDD 2017).
  [`PAMAE.scala`](spark/src/main/scala/clustering/algorithms/kmedoids/hybrid/PAMAE.scala)
- **`dbscanpp`** (alias **`dbscanexact`** at `coreSampleFraction: 1.0`) — DBSCAN++, distributed
  ε-neighbour counting + ε-graph, driver-local union-find.
  [`DBSCANpp.scala`](spark/src/main/scala/clustering/algorithms/dbscan/DBSCANpp.scala)

Evaluation metrics (`config.evaluation.metrics`):
[`SilhouetteEvaluator.scala`](spark/src/main/scala/clustering/evaluation/SilhouetteEvaluator.scala) ·
[`DaviesBouldinEvaluator.scala`](spark/src/main/scala/clustering/evaluation/DaviesBouldinEvaluator.scala) ·
[`CalinskiHarabaszEvaluator.scala`](spark/src/main/scala/clustering/evaluation/CalinskiHarabaszEvaluator.scala)

## Architecture

`BenchmarkRunner` consumes exactly one per-run JSON config and produces exactly one JSON
result file under `<outputDir>/<runId>.json`. Failure modes still write a result
(`status: "failed"` + `errorMessage`) and exit non-zero, so SLURM array jobs never
silently lose runs.

```mermaid
flowchart TD
  CONF[/"Plik konfiguracyjny<br/>(--config &lt;runId&gt;.json)"/] -->|"wczytanie konfiguracji (RunConfig.fromFile)"| JOB

  subgraph JOB ["1. Wykonanie zadania klasteryzacji (SparkClusteringJob)"]
    direction TB

    subgraph REG ["Odwzorowanie parametrów na obiekty (Spark Driver)"]
      direction TB
      DSR["Wybór źródła danych<br/>(DataSourceRegistry)"]
      ALGOR["Wybór algorytmu<br/>(AlgorithmRegistry)"]
      DIST["Wybór miary odległości<br/>(DistanceRegistry)"]

      ALGOR -.->|"pobiera metrykę odległości"| DIST
    end

    subgraph CLUSTER ["Rozproszone obliczenia (Spark executors)"]
      direction LR
      LOAD["Ładowanie danych<br/>(DataSource.load)"] --> FIT["Właściwa klasteryzacja<br/>(Clusterer.fit)"] --> EVAL["Ewaluacja wyników<br/>(EvaluationRunner.run)"]
    end

    REG -->|"przekazanie gotowych instancji"| LOAD
  end

  subgraph TELE ["Zbieranie metryk"]
    direction TB
    PCP["Zużycie zasobów JVM executora i drivera<br/>(ProcessCpuPlugin)"]
  end

  CLUSTER -.->|"próbkowanie JVM executorów (RPC)"| PCP

  CLUSTER -->|"koniec obliczeń"| STOP

  subgraph FINISH ["2. Agregacja i zakończenie"]
    direction TB
    STOP["Zamknięcie sesji, ostatnia próbka executorów<br/>(spark.stop())"]
    READ["Odczyt zebranych metryk"]
    RES["Scalenie metryk, wyników i czasów<br/>(RunResult.from)"]

    STOP --> READ
    READ --> RES
  end

  TELE -.->|"zrzuty metryk"| READ
  RES --> OUT[/"Plik wynikowy<br/>&lt;runId&gt;.json (ok | failed)"/]
```

The pipeline is assembled from config strings by three registries, so adding an
algorithm, data source, or metric is one factory entry — the runner never changes:

- **`AlgorithmRegistry`** — `config.algorithm.name` → `Clusterer` factory.
- **`DataSourceRegistry`** — `config.dataset.type` → `DataSource` factory (synthetic / Parquet).
- **`DistanceRegistry`** — `params.distance` → `DistanceMetric` (Euclidean / Manhattan / Cosine).

Core abstractions (`clustering.core`) keep algorithms uniform:

- `Clusterer.fit(DataFrame): Model` — the fitting seam every algorithm implements.
- `Model.assignClusters(DataFrame): DataFrame` — adds a `Columns.Prediction` column with
  each row's cluster id. Batch-only by design: some models (e.g. DBSCAN) have no
  meaningful per-point predict, so single-point prediction is not part of the contract.

`SparkClusteringJob` wires these together for one run: load the `DataSource`, `fit` the
`Clusterer`, evaluate the `Model`, and collect metrics into a `RunResult`.

## Project structure
```
.
├── spark/                        # Scala/Spark SBT project (the fat jar)
│   ├── build.sbt                 # Scala/Spark build, assembly into a fat jar
│   ├── src/main/scala/clustering/
│   │   ├── core/                 # Clusterer / Model / Geometry / Weights abstractions
│   │   ├── algorithms/
│   │   │   ├── kmeans/           # Lloyd, breathing refine, bisecting (hierarchical/)
│   │   │   ├── kmedoids/         # FastPAM/FasterPAM (local/), DistributedFastPAM (distributed/), CLARA/PAMAE (hybrid/)
│   │   │   └── dbscan/           # DBSCAN++ / exact DBSCAN (components/, utils/)
│   │   ├── distance/             # Euclidean / Manhattan / Cosine / UnitSphere metrics
│   │   ├── evaluation/           # Silhouette / Davies-Bouldin / Calinski-Harabasz
│   │   ├── utils/                # convergence, partition aggregation, union-find
│   │   └── benchmark/            # runner (--config) + config, datasource, evaluation runner, metrics, registry
│   └── ...
├── local_run.sh                  # spark-submit a single run locally
├── local_testing/                # JSON configs for local single-run testing
└── benchmark-results/            # local run outputs (one JSON result per run)
```

## Requirements

* JDK 11
* Scala 2.12 / sbt (with `sbt-assembly`)
* Apache Spark 3.3.2 (Hadoop 3)

## Usage

Build the fat jar:
```bash
cd spark
sbt assembly        # -> target/scala-2.12/spark-clustering-benchmark.jar
```

Run a single benchmark locally (from repo root):
```bash
./local_run.sh local_testing/experiment_configs/example.json
```

`local_run.sh` submits one config in `local` mode and writes the result to
`benchmark-results/`. Each config (dataset, algorithm, evaluation, Spark conf) conforms
to `run_config.schema.json` in the harness repo.

## Contract

The JSON exchanged between the repos is defined by
[`contract/README.md`](https://github.com/natix-x/clustering-algorithms-benchmark/blob/main/contract/README.md)
in the harness repo. This jar **consumes** `run_config.schema.json` (via
`--config <runId>.json`) and **produces** `run_result.schema.json`, so analysis is
uniform across engines.

Conformance is verified by a test, not at runtime. `SparkClusteringJobContractSpec`
runs a real `SparkClusteringJob` on a tiny synthetic dataset and asserts that the
emitted `RunResult` validates against the vendored `run_result.schema.json`
(`spark/src/test/resources/`). It covers both an `ok` run and a `failed` run
(unknown algorithm), and because the schema is `additionalProperties: false`, any
Scala field not declared in the contract makes the test fail — guarding against
schema drift.

```bash
cd spark
sbt "testOnly clustering.benchmark.SparkClusteringJobContractSpec"
```
