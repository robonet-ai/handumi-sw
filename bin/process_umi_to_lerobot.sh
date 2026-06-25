#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bin/process_umi_to_lerobot.sh  –  launcher for scripts/process_umi_to_lerobot.py
#
# Converts a PICO/UMI LeRobot dataset to embodiment-specific joint angles via IK.
#
# Usage (all arguments are optional; defaults shown below):
#
#   bash bin/process_umi_to_lerobot.sh \
#       --repo-id NONHUMAN-RESEARCH/dexumi-dataset-v2 \
#       --dataset-root outputs/datasets/dexumi-dataset-v2 \
#       --embodiment piper \
#       --output-name dexumi-dataset-v2-piper \
#       --output-root outputs/datasets/dexumi-dataset-v2-piper
#
# Extra flags:
#   --push-to-hub          Upload to HuggingFace Hub after processing
#   --episodes 0,1,2       Process only selected episode indices
#   --task "Pick cube"     Override task description for all episodes
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve workspace root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Activate the virtual environment ──────────────────────────────────────────
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    echo "[process_umi_to_lerobot.sh] Virtual environment activated: ${VENV}"
else
    echo "[process_umi_to_lerobot.sh] WARNING: no .venv found at ${VENV}. Using system Python."
fi

# ── Make the dexumi package importable ────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ── Default arguments (override via CLI or env vars) ──────────────────────────
REPO_ID="${REPO_ID:-NONHUMAN-RESEARCH/dexumi-dataset-v2}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/datasets/dexumi-dataset-v2}"
REVISION="${REVISION:-main}"
EMBODIMENT="${EMBODIMENT:-piper}"
OUTPUT_NAME="${OUTPUT_NAME:-dexumi-dataset-v2-piper}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-NONHUMAN-RESEARCH/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/datasets/dexumi-dataset-v2-piper}"
EPISODES="${EPISODES:-}"
TASK="${TASK:-}"

# ── Parse CLI arguments ───────────────────────────────────────────────────────
EXTRA_FLAGS=()
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)        REPO_ID="$2";        shift 2 ;;
        --dataset-root)   DATASET_ROOT="$2";   shift 2 ;;
        --revision)       REVISION="$2";       shift 2 ;;
        --embodiment)     EMBODIMENT="$2";     shift 2 ;;
        --output-name)    OUTPUT_NAME="$2";    shift 2 ;;
        --output-prefix)  OUTPUT_PREFIX="$2";  shift 2 ;;
        --output-root)    OUTPUT_ROOT="$2";    shift 2 ;;
        --episodes)       EPISODES="$2";       shift 2 ;;
        --task)           TASK="$2";           shift 2 ;;
        --push-to-hub)
            EXTRA_FLAGS+=("$1"); shift ;;
        *)
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Print configuration ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     dexumi  –  UMI → LeRobot embodiment conversion       ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Source repo   : %-40s║\n" "${REPO_ID}"
printf "║  Dataset root  : %-40s║\n" "${DATASET_ROOT}"
printf "║  Revision      : %-40s║\n" "${REVISION}"
printf "║  Embodiment    : %-40s║\n" "${EMBODIMENT}"
printf "║  Output name   : %-40s║\n" "${OUTPUT_NAME}"
printf "║  Output root   : %-40s║\n" "${OUTPUT_ROOT}"
printf "║  Episodes      : %-40s║\n" "${EPISODES:-all}"
printf "║  Task override : %-40s║\n" "${TASK:-none}"
printf "║  Extra flags   : %-40s║\n" "${EXTRA_FLAGS[*]:-none}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

CMD=(
    python "${REPO_ROOT}/scripts/process_umi_to_lerobot.py"
    --repo-id "${REPO_ID}"
    --dataset-root "${DATASET_ROOT}"
    --revision "${REVISION}"
    --embodiment "${EMBODIMENT}"
    --output-name "${OUTPUT_NAME}"
    --output-prefix "${OUTPUT_PREFIX}"
    --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${EPISODES}" ]]; then
    CMD+=(--episodes "${EPISODES}")
fi

if [[ -n "${TASK}" ]]; then
    CMD+=(--task "${TASK}")
fi

CMD+=("${EXTRA_FLAGS[@]}" "${PASSTHROUGH[@]}")

exec "${CMD[@]}"
