#!/usr/bin/env bash
#
# Run one experiment. Defaults come from configs.yaml; any extra argument is
# passed straight through to the experiment.
#
#     ./run.sh preprocess
#     ./run.sh classification --epochs 50 --ragas kalyani sahana
#     ./run.sh figures --list
#
set -euo pipefail

if [[ $# -eq 0 ]]; then
    exec python -m rise --help
fi

exec python -m rise "$@"
