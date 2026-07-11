#!/bin/bash -l

set -euo pipefail

# Resolve paths relative to this script so it works from any cwd.
# Script lives at repo root; the sbt build root is the nested spark/ dir.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_DIR="${REPO_ROOT}/spark"

CONFIG_PATH="${1:?usage: $0 <path-to-per-run-config.json>}"

: "${SPARK_HOME:=$HOME/spark-local/spark-3.3.2-bin-hadoop3}"
: "${JAR_PATH:=${SPARK_DIR}/target/scala-2.12/spark-clustering-benchmark.jar}"
: "${JAVA_HOME:=/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home}"
: "${DRIVER_MEM:=2g}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

# Bind Spark to loopback so a VPN/proxy on the LAN interface can't intercept
# the driver's internal RPC/file-server connections ("Too large frame" error).
export SPARK_LOCAL_IP=127.0.0.1

# Run from repo root so LocalProfile's default outputDir (user.dir/benchmark-results)
# lands in the repo-root benchmark-results/ dir.
cd "${REPO_ROOT}"

"${SPARK_HOME}/bin/spark-submit" \
  --master "local[*]" \
  --driver-memory "${DRIVER_MEM}" \
  --conf "spark.driver.bindAddress=127.0.0.1" \
  --conf "spark.driver.host=127.0.0.1" \
  --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
  --conf "spark.driver.extraJavaOptions=-Dlog4j2.configurationFile=${SPARK_DIR}/src/main/resources/log4j2.properties" \
  "${JAR_PATH}" \
  --config "${CONFIG_PATH}"