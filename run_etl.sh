#!/usr/bin/env bash
set -euo pipefail

# load .env safely
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

: "${DSN:?DSN variable is required in .env}"
DATA_DIR="${DATA_DIR:-./data}"
SCHEMA_FILE="${SCHEMA_FILE:-schema.sql}"

exec python3 main.py \
  --dsn "$DSN" \
  --data-dir "$DATA_DIR" \
  --schema-file "$SCHEMA_FILE" \
  --once \
  --refresh-mv \
  --log-level INFO
