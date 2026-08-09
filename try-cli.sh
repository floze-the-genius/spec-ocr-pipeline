#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHONPATH=src python3 -m spec_pipeline.cli setup-windows "$@"
