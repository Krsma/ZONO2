#!/usr/bin/env bash
# Start Jupyter with persistent student files and localhost-only access.
set -euo pipefail
exec docker run --rm --init -it \
  -p "127.0.0.1:${PZR_STUDENT_PORT:-8888}:8888" \
  -v "${PZR_STUDENT_VOLUME:-pzr-student-workspace}:/workspace" \
  "${PZR_STUDENT_IMAGE:-pzr-student:local}" "$@"
