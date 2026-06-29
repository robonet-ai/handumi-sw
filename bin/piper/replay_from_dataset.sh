#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# bin/piper/replay_from_dataset.sh
#
# Launcher for scripts/piper/replay_from_dataset.py.
#
# Paso 2 - CAN (mapear, activar, verificar)
#
# Identificar que bus corresponde a cada brazo:
#
#   sudo ethtool -i can0 | grep bus-info   # ej. 1-1.2:1.0 (brazo izq)
#   sudo ethtool -i can1 | grep bus-info   # ej. 1-6.2:1.0 (brazo der)
#
# Activar cada bus (reemplaza <BUS-ID> con el valor anterior):
#
#   sudo bash ~/miniconda3/envs/xhuman/lib/python3.11/site-packages/piper_sdk/can_activate.sh can0 1000000 1-1:1.0
#   sudo bash ~/miniconda3/envs/xhuman/lib/python3.11/site-packages/piper_sdk/can_activate.sh can1 1000000 1-2:1.0
#
# Confirmar que estan state UP:
#
#   ip link show can0 | grep state
#   ip link show can1 | grep state
#
# Ejemplos:
#
#   # Validar unidades/conversiones sin conectar al robot
#   bash bin/piper/replay_from_dataset.sh --dry-run --frames 5
#
#   # Ejecutar en ambos brazos, con confirmacion interactiva
#   bash bin/piper/replay_from_dataset.sh --episode 0
#
#   # Ejecutar sin prompt interactivo
#   bash bin/piper/replay_from_dataset.sh --episode 0 --yes
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    echo "[replay_from_dataset.sh] Virtual environment activated: ${VENV}"
else
    echo "[replay_from_dataset.sh] WARNING: no .venv found at ${VENV}. Using system Python."
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export LEFT_ROBOT_ARM_PORT="${LEFT_ROBOT_ARM_PORT:-can0}"
export RIGHT_ROBOT_ARM_PORT="${RIGHT_ROBOT_ARM_PORT:-can1}"

REPO_ID="${REPO_ID:-NONHUMAN-RESEARCH/handumi-dataset-v2-piper}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/datasets/handumi-dataset-v2-piper}"
EPISODE="${EPISODE:-0}"
ARMS="${ARMS:-both}"
SPEED_PERCENT="${SPEED_PERCENT:-30}"

echo ""
echo "----------------------------------------------------------"
echo " handumi - Piper dataset trajectory replay"
echo "----------------------------------------------------------"
printf " Repo id      : %s\n" "${REPO_ID}"
printf " Dataset root : %s\n" "${DATASET_ROOT}"
printf " Episode      : %s\n" "${EPISODE}"
printf " Arms         : %s\n" "${ARMS}"
printf " Left CAN     : %s\n" "${LEFT_ROBOT_ARM_PORT}"
printf " Right CAN    : %s\n" "${RIGHT_ROBOT_ARM_PORT}"
printf " Speed        : %s %%\n" "${SPEED_PERCENT}"
echo "----------------------------------------------------------"
echo ""

exec python "${REPO_ROOT}/scripts/piper/replay_from_dataset.py" \
    --repo-id "${REPO_ID}" \
    --dataset-root "${DATASET_ROOT}" \
    --episode "${EPISODE}" \
    --arms "${ARMS}" \
    --left-port "${LEFT_ROBOT_ARM_PORT}" \
    --right-port "${RIGHT_ROBOT_ARM_PORT}" \
    --speed-percent "${SPEED_PERCENT}" \
    "$@"
