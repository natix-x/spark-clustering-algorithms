"""
Template for SLURM job script.
"""

SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH -J {name}-N{nodes}-{run_id} 
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={executors_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --exclusive
#SBATCH --time={walltime}
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH --output={log_dir}/{run_id}_%j.out
#SBATCH --error={log_dir}/{run_id}_%j.err
 
set -euo pipefail
module purge && module load {spark_module}
 
CONFIG_PATH="{config_path}"
RUN_ID="{run_id}"
 
if [ ! -f "$CONFIG_PATH" ]; then
    echo "$CONFIG_PATH not found!"
    exit 1
fi
 
echo "=== RUN: $RUN_ID  (Job=$SLURM_JOB_ID) ==="
 
# Izolated worker dirs - generated per run
export SPARK_LOG_DIR="{output_dir}/spark_logs/${{SLURM_JOB_ID}}_{run_id}"
export SPARK_WORKER_DIR="/tmp/spark_work_${{SLURM_JOB_ID}}_{run_id}"
mkdir -p "$SPARK_LOG_DIR" "$SPARK_WORKER_DIR"
 
# Spark config
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
 
MASTER_PORT=$(python3 -c "
import hashlib, sys
seed = hashlib.sha256(sys.argv[1].encode()).digest()
print(49152 + int.from_bytes(seed[:2], 'big') % 16383) 
" "{run_id}")  # Choosing a port in the range [49152, 65535]
 
# Exporting Spark config 
export SPARK_MASTER_HOST="$MASTER_HOST"
export SPARK_MASTER_PORT="$MASTER_PORT"
export SPARK_MASTER_WEBUI_PORT=$(( 8080 + SLURM_JOB_ID % 1000 ))
export SPARK_WORKER_WEBUI_PORT=$(( 8081 + SLURM_JOB_ID % 1000 ))
export SPARK_WORKER_MEMORY="{worker_mem}g"
export SPARK_WORKER_CORES="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
 
MASTER_URL="spark://$MASTER_HOST:$MASTER_PORT"
MASTER_UI="http://$MASTER_HOST:$SPARK_MASTER_WEBUI_PORT"
CORES="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"
EXECUTORS_PER_NODE={executors_per_node}
TOTAL_EXECUTORS=$(( SLURM_NNODES * EXECUTORS_PER_NODE ))
 
echo "Config:          $CONFIG_PATH"
echo "Master URL:      $MASTER_URL"
echo "Master Web UI:   $MASTER_UI"
echo "Nodes:           $SLURM_NNODES  (NODELIST=$SLURM_JOB_NODELIST)"
echo "Executors:       $TOTAL_EXECUTORS  ($EXECUTORS_PER_NODE/nodes x $CORES cores)"
echo "Executor memory: {executor_mem}g   driver: {driver_mem}g"
 
# Clean up on every exit
trap '
  EXIT_CODE=$?
  echo ">>> Trap EXIT (kod=$EXIT_CODE) — cleaning up {run_id}"
  SPARK_IDENT_STRING="${{SLURM_JOB_ID}}_{run_id}" \
      "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
  rm -rf "$SPARK_WORKER_DIR"
  exit $EXIT_CODE
' EXIT INT TERM
 
# Start Master
"$SPARK_HOME/sbin/start-master.sh"
 
echo -n "Waiting for Master to start"
MASTER_READY=""
_i=0
while [ "$_i" -lt 40 ]; do
    _i=$(( _i + 1 ))
    sleep 2
    LOG_FILE=$(find "$SPARK_LOG_DIR" -name "*.out" 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ] && grep -q "I have been elected leader" "$LOG_FILE" 2>/dev/null; then
        MASTER_READY="yes"
        echo " OK (${{_i}} s)"
        break
    fi
    echo -n "."
done
if [ -z "$MASTER_READY" ]; then
    echo ""
    echo "ERROR: Master did not start in 80s. Logs in  $SPARK_LOG_DIR:"
    find "$SPARK_LOG_DIR" -name "*.out" 2>/dev/null | while read -r f; do
        echo "--- $f ---"; tail -5 "$f"
    done
    exit 1
fi
 
# Start workers
srun --ntasks-per-node="$EXECUTORS_PER_NODE" \
     --cpus-per-task="$CORES" \
     --export=ALL \
     --output="$SPARK_LOG_DIR/workers-%j-%t.out" \
     "$SPARK_HOME/bin/spark-class" \
     org.apache.spark.deploy.worker.Worker \
     --cores "$CORES" \
    --memory "$SPARK_WORKER_MEMORY" \
     "$MASTER_URL" &
 
echo -n "Waiting for $TOTAL_EXECUTORS workers (REST $MASTER_UI/json/)"
WORKERS_READY=0

for _i in $(seq 1 60); do
    WORKERS_READY=$(curl -sf "$MASTER_UI/json/" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('aliveworkers', 0))" \
    2>/dev/null || echo 0)
    
    if [ "$WORKERS_READY" -ge "$TOTAL_EXECUTORS" ]; then
        echo " OK ($WORKERS_READY/$TOTAL_EXECUTORS w ${{_i}} s)"
        break
    fi
    echo -n "."
    sleep 2
done

if [ "$WORKERS_READY" -lt "$TOTAL_EXECUTORS" ]; then
    echo -e "\nERROR: Only $WORKERS_READY/$TOTAL_EXECUTORS workers ready. Stopping."
    echo "Cluster status (API):"
    curl -sf "$MASTER_UI/json/" 2>/dev/null \
    | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" \
    2>/dev/null || echo "(API niedostepne)"
    
    exit 1
fi
 
# Submit job
spark-submit \
  --master "$MASTER_URL" \
  --deploy-mode client \
  --driver-memory "{driver_mem}g" \
  --executor-cores "$CORES" \
  --executor-memory "{executor_mem}g" \
  --num-executors "$TOTAL_EXECUTORS" \
  "{jar}" --config "$CONFIG_PATH"
"""

# TODO: refactor this script - division into smaller functions, make it more readable