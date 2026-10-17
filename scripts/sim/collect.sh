#!/usr/bin/env bash
# RoboTwin2 expert data collection in the r3d_sim env (CLAUDE.md "Sim baseline").
#
#   bash scripts/sim/collect.sh place_empty_cup_right sim_head_p1024
#
# Output: R3D/r3d/env/robotwin2/data/<task>/<task_config>/data/episode*.hdf5,
# then convert with scripts/convert_robotwin_hdf5_to_zarr.py.
#
# RoboTwin imports envs/ both as a package and flat (`from _GLOBAL_CONFIGS import`
# in envs/utils/get_camera_config.py), so envs/ must be on PYTHONPATH. The ROS
# PYTHONPATH is dropped first (see CLAUDE.md "Environment"); only repo dirs remain.
set -euo pipefail
task=${1:?usage: collect.sh <task> <task_config>}
task_config=${2:?missing task_config}
rt="$(cd "$(dirname "$0")/../../R3D/r3d/env/robotwin2" && pwd)"
eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
cd "${rt}"
PYTHONPATH="${rt}:${rt}/envs:${rt}/description/utils" \
    python -W ignore script/collect_data.py "${task}" "${task_config}"
