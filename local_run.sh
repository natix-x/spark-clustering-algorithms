#!/bin/bash -l

set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:?usage: $0 <path-to-per-run-config.json>}"

: "${SPARK_HOME:=$HOME/spark-local/spark-3.3.2-bin-hadoop3}"
: "${JAR_PATH:=${SCRIPT_DIR}/target/scala-2.12/spark-clustering-benchmark.jar}"
: "${JAVA_HOME:=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home}"
: "${DRIVER_MEM:=2g}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

# Run from repo root so LocalProfile's default outputDir (user.dir/benchmark-results)
# lands in the repo-root benchmark-results/ dir.
cd "${REPO_ROOT}"

"${SPARK_HOME}/bin/spark-submit" \
  --master "local[*]" \
  --driver-memory "${DRIVER_MEM}" \
  --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
  --conf "spark.driver.extraJavaOptions=-Dlog4j2.configurationFile=${SCRIPT_DIR}/src/main/resources/log4j2.properties" \
  "${JAR_PATH}" \
  --config "${CONFIG_PATH}"