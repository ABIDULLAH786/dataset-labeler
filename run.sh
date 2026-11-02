#!/usr/bin/env bash
# Convenience launcher. Edit ROOT and MODEL for your machine.
set -euo pipefail
ROOT="${1:-$HOME/Downloads/Sampel files/drone-datasets/New/labeled}"
MODEL="${MODEL:-}"
ARGS=(--root "$ROOT" --port "${PORT:-8000}")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
exec python3 server.py "${ARGS[@]}"
