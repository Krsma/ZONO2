#!/usr/bin/env bash
# Start Jupyter with persistent learning files and localhost-only access.
set -euo pipefail
exec docker run --rm --init -it \
  -p "127.0.0.1:${PZR_LEARNING_PORT:-8888}:8888" \
  -v "${PZR_LEARNING_VOLUME:-pzr-learning-workspace}:/workspace" \
  "${PZR_LEARNING_IMAGE:-pzr-learning:local}" "$@"
