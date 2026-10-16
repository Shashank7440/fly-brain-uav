#!/bin/bash
# CPU-only installer/checks. Does not submit jobs or replace model/config files.
set -eo pipefail
ROOT=/Data4/mi_25026207/drone/v8_wireflynet
cd "$ROOT"
PKG="$ROOT/v8_weighted_controls_package"
mkdir -p jobs tools

copy_new() {
    if [[ -e "$2" ]]; then
        if ! cmp -s "$1" "$2"; then
            echo "Different existing destination: $2. Stop; it was not overwritten."
            exit 1
        fi
    else
        cp "$1" "$2"
    fi
}

copy_new "$PKG/jobs/v8_weighted_controls.pbs" "$ROOT/jobs/v8_weighted_controls.pbs"
copy_new "$PKG/tools/preflight_weighted_controls.py" "$ROOT/tools/preflight_weighted_controls.py"
copy_new "$PKG/tools/submit_weighted_controls.py" "$ROOT/tools/submit_weighted_controls.py"
python3 "$PKG/tests/test_control_package.py"
python3 -m py_compile tools/preflight_weighted_controls.py tools/submit_weighted_controls.py
bash -n jobs/v8_weighted_controls.pbs
python3 tools/preflight_weighted_controls.py
echo "INSTALL/CHECK COMPLETE. No jobs submitted."