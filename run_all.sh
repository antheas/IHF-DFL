#!/usr/bin/env bash
# Runs every experiment behind the FAIRFL paper's result figures (../polfl) and regenerates the plots.
#
# Grid (paper Tables "static parameters" / "hyperparameter grid"):
#   {naive, loss_only, accuracy_only, accuracy_loss} x free-rider noise {0.01, 0.1}
#   x free-rider activation round {1, 3, 5} x 10 repetitions, on MNIST and CIFAR-10 = 480 runs.
# These are exactly the presets {mnist,cifar}_openfl_{low,high}_noise in experiment/experiment_presets.py.
#
# Re-running resumes: runs that already finished (have a .pkl) in the result folder are skipped.
#
# Usage: ./run_all.sh [--plot-only]
# Environment overrides:
#   PYTHON      python interpreter          (default: .venv/bin/python)
#   PAPER_DIR   paper repository            (default: ../polfl)
#   RUN_NAME    result subfolder name       (default: paper)
#   PARALLEL    runs at the same time       (default: 1; CIFAR runs need a lot of GPU memory each)
#   AUTHOR      author written to the CSVs  (default: $USER)
#   RPC_PORT    first Ganache port; PARALLEL=N uses N nodes on RPC_PORT..RPC_PORT+N-1 (default: 7545)
#   FIG_DIR     where figures/summaries go  (default: data/figures/$RUN_NAME;
#               use $PAPER_DIR/figures/results to overwrite the paper's figures)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/../polfl}"
RUN_NAME="${RUN_NAME:-paper}"
PARALLEL="${PARALLEL:-1}"
AUTHOR="${AUTHOR:-${USER:-unknown}}"
RPC_PORT="${RPC_PORT:-7545}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/data/figures/$RUN_NAME}"

RESULT_DIR="$REPO_ROOT/data/runs/experiments/$RUN_NAME"
CSV_DIR="$REPO_ROOT/data/runs/paper_csv/$RUN_NAME"
PLOT_SCRIPT="$PAPER_DIR/analysis/plot_results.py"
EXPECTED_RUNS=480

# preset:dataset (dataset names as experiment/experiments.py expects them)
EXPERIMENTS=(
  "mnist_openfl_low_noise:mnist"
  "mnist_openfl_high_noise:mnist"
)
  # "cifar_openfl_low_noise:cifar.10"
  # "cifar_openfl_high_noise:cifar.10"

PLOT_ONLY=0
case "${1:-}" in
  "") ;;
  --plot-only) PLOT_ONLY=1 ;;
  *) echo "Usage: $0 [--plot-only]" >&2; exit 2 ;;
esac

log() { echo "[$(date '+%F %T')] $*"; }

[[ -x "$PYTHON" ]] || { echo "Python not found at $PYTHON (set PYTHON=...)" >&2; exit 1; }
[[ -f "$PLOT_SCRIPT" ]] || { echo "Plot script not found at $PLOT_SCRIPT (set PAPER_DIR=...)" >&2; exit 1; }

# ---------------- GANACHE ----------------

# One node per parallel run: runs sharing a node collide on account nonces and on the chain clock.
GANACHE_PIDS=()
RPC_URLS=""

rpc_up() {
  curl -s -m 2 -X POST -H 'Content-Type: application/json' \
    --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
    "http://127.0.0.1:$1" >/dev/null 2>&1
}

start_ganache() {
  command -v ganache >/dev/null || { echo "ganache not found; install it (npm i -g ganache)" >&2; exit 1; }
  mkdir -p "$RESULT_DIR"
  local urls=() port pid
  for ((i = 0; i < PARALLEL; i++)); do
    port=$((RPC_PORT + i))
    if rpc_up "$port"; then
      echo "Port $port is already in use by another node; stop it or set RPC_PORT=..." >&2
      exit 1
    fi
    # Many accounts with large balances and a high gas limit (see README "Ganache").
    ganache --server.port "$port" \
      --wallet.totalAccounts 10 \
      --wallet.defaultBalance 1000000 \
      --miner.blockGasLimit 1000000000 \
      --logging.quiet >>"$RESULT_DIR/ganache-$port.log" 2>&1 &
    GANACHE_PIDS+=($!)
    urls+=("http://127.0.0.1:$port")
  done
  for ((i = 0; i < PARALLEL; i++)); do
    port=$((RPC_PORT + i))
    pid="${GANACHE_PIDS[$i]}"
    for _ in $(seq 60); do
      rpc_up "$port" && break
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    rpc_up "$port" || { echo "ganache did not come up, see $RESULT_DIR/ganache-$port.log" >&2; exit 1; }
  done
  RPC_URLS="$(IFS=,; echo "${urls[*]}")"
  log "Started $PARALLEL ganache node(s): $RPC_URLS"
}

stop_ganache() {
  local pid
  for pid in "${GANACHE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  GANACHE_PIDS=()
}
trap stop_ganache EXIT

# ---------------- EXPERIMENTS ----------------

if [[ $PLOT_ONLY -eq 0 ]]; then
  log "Compiling contracts"
  "$PYTHON" scripts/compile_contracts.py

  for entry in "${EXPERIMENTS[@]}"; do
    preset="${entry%%:*}"
    dataset="${entry##*:}"
    # A fresh chain per preset keeps ganache's in-memory state from growing over all 480 runs.
    start_ganache
    log "Running preset $preset on $dataset -> $RESULT_DIR/$preset"
    # RPC_URL (sequential runs) and RPC_URLS (parallel runs) take precedence over .env/.env.ganache.
    RPC_URL="${RPC_URLS%%,*}" RPC_URLS="$RPC_URLS" "$PYTHON" experiment/experiments.py \
      --preset "$preset" \
      --dataset "$dataset" \
      --outputName "$RUN_NAME/$preset" \
      --author "$AUTHOR" \
      --parallel "$PARALLEL"
    stop_ganache
  done
fi

# ---------------- COLLECT ----------------

# Only finished runs have a .pkl; a failed run leaves a partial .csv (and an error-*.txt) behind.
log "Collecting finished runs into $CSV_DIR"
rm -rf "$CSV_DIR"
mkdir -p "$CSV_DIR"
found=0
if [[ -d "$RESULT_DIR" ]]; then
  while IFS= read -r -d '' csv; do
    if [[ -f "${csv%.csv}.pkl" ]]; then
      cp "$csv" "$CSV_DIR/"
      found=$((found + 1))
    fi
  done < <(find "$RESULT_DIR" -name '*.csv' -print0)
fi
log "Found $found/$EXPECTED_RUNS finished runs"

if [[ $found -ne $EXPECTED_RUNS ]]; then
  errors=$(find "$RESULT_DIR" -name 'error-*.txt' 2>/dev/null | wc -l)
  echo "Missing runs ($errors error file(s) under $RESULT_DIR). Re-run $0 to retry the missing ones." >&2
  exit 1
fi

# ---------------- PLOT ----------------

log "Generating figures into $FIG_DIR"
"$PYTHON" "$PLOT_SCRIPT" --input "$CSV_DIR" --output "$FIG_DIR"
log "Done"
