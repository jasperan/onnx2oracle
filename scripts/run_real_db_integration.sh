#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/src/onnx2oracle/data/docker-compose.yml"
LOG_DIR="${ONNX2ORACLE_LOG_DIR:-$ROOT/integration-artifacts}"
WAIT_TIMEOUT="${ONNX2ORACLE_WAIT_TIMEOUT:-900}"
CLEANUP="${ONNX2ORACLE_CLEANUP:-keep}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/real-db-$(date -u +%Y%m%dT%H%M%SZ).log"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export ORACLE_PWD="${ORACLE_PWD:-onnx2oracle}"
export ORACLE_PORT="${ORACLE_PORT:-1521}"
export ORACLE_DSN="system/${ORACLE_PWD}@localhost:${ORACLE_PORT}/FREEPDB1"
SAFE_ORACLE_TARGET="$(
  python - <<'PY'
import os
from onnx2oracle.connection import DSN

try:
    print(DSN.parse(os.environ["ORACLE_DSN"]).display())
except Exception:
    print("<redacted ORACLE_DSN>")
PY
)"

run() {
  printf '+ %q' "$@" | tee -a "$LOG_FILE"
  printf '\n' | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_sql() {
  local label="$1"
  local sql="$2"
  printf '+ sqlplus %s\n' "$label" | tee -a "$LOG_FILE"
  docker compose -f "$COMPOSE_FILE" exec -T oracle bash -lc \
    'sqlplus -L -S "system/${ORACLE_PWD:-onnx2oracle}@localhost:1521/FREEPDB1"' <<SQL | tee -a "$LOG_FILE"
set heading off feedback off pagesize 200 linesize 240 trimspool on
$sql
exit
SQL
}

run_compose_summary() {
  printf '+ docker compose -f %q config --services\n' "$COMPOSE_FILE" | tee -a "$LOG_FILE"
  docker compose -f "$COMPOSE_FILE" config --services 2>&1 | tee -a "$LOG_FILE"
  printf '+ docker compose -f %q config --images\n' "$COMPOSE_FILE" | tee -a "$LOG_FILE"
  docker compose -f "$COMPOSE_FILE" config --images 2>&1 | tee -a "$LOG_FILE"
}

cleanup() {
  case "$CLEANUP" in
    keep)
      ;;
    down)
      run python -m onnx2oracle.cli docker down
      ;;
    volumes)
      run python -m onnx2oracle.cli docker down --volumes
      ;;
    *)
      printf 'Unknown ONNX2ORACLE_CLEANUP=%s (expected keep, down, or volumes)\n' "$CLEANUP" | tee -a "$LOG_FILE"
      return 1
      ;;
  esac
}
trap cleanup EXIT

{
  printf 'onnx2oracle real DB integration evidence\n'
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'root=%s\n' "$ROOT"
  printf 'oracle_image=%s\n' "${ORACLE_IMAGE:-container-registry.oracle.com/database/free:latest}"
  printf 'oracle_port=%s\n' "$ORACLE_PORT"
  printf 'oracle_target=%s\n' "$SAFE_ORACLE_TARGET"
  printf 'cleanup=%s\n' "$CLEANUP"
} | tee "$LOG_FILE"

run docker version
run_compose_summary
run python -m onnx2oracle.cli docker up --wait --wait-timeout "$WAIT_TIMEOUT"
run docker inspect onnx2oracle-oracle --format 'image={{.Config.Image}} image_id={{.Image}}'

run_sql "database banner" "SELECT banner FROM v\$version FETCH FIRST 1 ROW ONLY;"
run_sql "DBMS_VECTOR package" "SELECT owner || '.' || object_name || ':' || status FROM all_objects WHERE object_name = 'DBMS_VECTOR' AND object_type = 'PACKAGE';"
run_sql "CREATE MINING MODEL privilege" "SELECT privilege FROM session_privs WHERE privilege = 'CREATE MINING MODEL';"

run python -m onnx2oracle.cli preflight --target local
run python -m onnx2oracle.cli load all-MiniLM-L6-v2 --target local --force
run python -m onnx2oracle.cli verify --target local --name ALL_MINILM_L6_V2
run python -m pytest tests/test_loader_integration.py --run-integration -v -s

printf 'Evidence log: %s\n' "$LOG_FILE" | tee -a "$LOG_FILE"
