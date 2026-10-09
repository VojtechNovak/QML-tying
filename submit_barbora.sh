#!/usr/bin/env bash
# One public entry point: installs into an existing env and submits one serial job.
set -Eeuo pipefail
QML_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 MPLBACKEND=Agg

if [[ ${1:-} == --batch ]]; then
    QML_PYTHON=$2
    QML_SETTINGS=$3
    cd -- "$QML_PROJECT_DIR"
    QML_OUT=$("$QML_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["out"])' "$QML_SETTINGS")
    QML_PID=
    qml_stop() { [[ -z $QML_PID ]] || kill -USR1 "$QML_PID" 2>/dev/null || true; }
    qml_finalize() {
        QML_STATUS=$?
        trap - EXIT USR1 TERM INT
        printf '%s\n' "$QML_STATUS" > "$QML_OUT/submission/batch_exit_code_${SLURM_JOB_ID:-local}.txt"
        # Analysis runs in the controller. This trap also packages failures.
        timeout --signal=TERM --kill-after=10s 180s "$QML_PYTHON" "$QML_PROJECT_DIR/collect_results.py" --out "$QML_OUT" --quick || true
        exit "$QML_STATUS"
    }
    trap qml_stop USR1 TERM INT
    trap qml_finalize EXIT
    "$QML_PYTHON" -u "$QML_PROJECT_DIR/cluster_run.py" --settings "$QML_SETTINGS" &
    QML_PID=$!
    QML_STATUS=0
    # A trapped warning interrupts wait; keep waiting for graceful finalization.
    while true; do
        if wait "$QML_PID"; then QML_STATUS=0; break; else QML_STATUS=$?; fi
        kill -0 "$QML_PID" 2>/dev/null || break
    done
    exit "$QML_STATUS"
fi

QML_ENV=${VIRTUAL_ENV:-${CONDA_PREFIX:-}}
QML_SKIP_INSTALL=0
QML_DRY_RUN=0
QML_COLLECT=
QML_MODULES=()
QML_FORWARD=()
while (($#)); do
    case "$1" in
        --env) QML_ENV=$2; shift 2 ;;
        --module) QML_MODULES+=("$2"); shift 2 ;;
        --skip-install) QML_SKIP_INSTALL=1; shift ;;
        --dry-run) QML_DRY_RUN=1; QML_FORWARD+=("$1"); shift ;;
        --collect) QML_COLLECT=$2; shift 2 ;;
        --help|-h)
            cat <<'HELP'
Usage: bash submit_barbora.sh --account PROJECT-ID [--env /existing/env] [options]
Activate your existing venv/conda env first, or specify its directory with --env.
Options: --partition qcpu (default), --hours H (default: up to 48, within remaining budget), --config FILE, --out DIR,
         --module NAME (repeatable), --skip-install, --skip-tests, --dry-run, --analysis-only,
         --budget-ledger /path/to/earlier/cluster_control/budget_ledger.json.
Collect after completion: bash submit_barbora.sh --collect results/barbora
Requirements are installed on the login node; all scientific work runs inside one Slurm allocation.
The cumulative ledger shares the previous sibling study's budget when present.
Pending/running prior jobs hold their reservation; nothing is cancelled automatically.
HELP
            exit 0 ;;
        *) QML_FORWARD+=("$1"); shift ;;
    esac
done
if ((${#QML_MODULES[@]})); then
    if ! type module &>/dev/null; then
        [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
    fi
    type module &>/dev/null || { echo 'module command unavailable; load your environment modules first.' >&2; exit 1; }
    for QML_MODULE in "${QML_MODULES[@]}"; do module load "$QML_MODULE"; done
fi
if [[ -n $QML_COLLECT && -z $QML_ENV ]]; then
    QML_PYTHON=$(python3 "$QML_PROJECT_DIR/cluster_submit.py" --print-python)
elif [[ -n $QML_ENV ]]; then
    # Keep the venv interpreter symlink path: realpath would lose the venv prefix.
    QML_ENV=$(cd -- "$QML_ENV" && pwd)
    QML_PYTHON=$QML_ENV/bin/python
    export PATH="$QML_ENV/bin:$PATH"
else
    echo 'Activate your existing Python environment or pass --env /path/to/existing/env.' >&2
    exit 1
fi
[[ -x $QML_PYTHON ]] || { echo "Missing interpreter: $QML_PYTHON" >&2; exit 1; }
if [[ -n $QML_COLLECT ]]; then
    exec "$QML_PYTHON" "$QML_PROJECT_DIR/cluster_submit.py" --python "$QML_PYTHON" --collect "$QML_COLLECT"
fi
"$QML_PYTHON" - <<'PY'
import os,sys
from pathlib import Path
if sys.version_info < (3,11):
    raise SystemExit('This package requires Python >=3.11 in the existing environment.')
if not (sys.prefix != sys.base_prefix or (Path(sys.prefix)/'conda-meta').is_dir()):
    raise SystemExit('The selected Python is not a venv/conda environment. Activate your existing environment or pass its prefix with --env.')
print('Existing environment:',sys.prefix,'Python',sys.version.split()[0])
PY
if (( !QML_SKIP_INSTALL && !QML_DRY_RUN )); then
    # Check the shared reservation BEFORE changing the existing environment.
    "$QML_PYTHON" "$QML_PROJECT_DIR/cluster_submit.py" --python "$QML_PYTHON" --dry-run "${QML_FORWARD[@]}" >/dev/null
    "$QML_PYTHON" -m pip install --only-binary=:all: -r "$QML_PROJECT_DIR/requirements.txt"
fi
exec "$QML_PYTHON" "$QML_PROJECT_DIR/cluster_submit.py" --python "$QML_PYTHON" "${QML_FORWARD[@]}"
