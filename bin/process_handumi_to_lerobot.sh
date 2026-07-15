#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bin/process_handumi_to_lerobot.sh  –  compatibility launcher for handumi-convert
#
# Converts a PICO/HandUMI LeRobot dataset to embodiment-specific joint angles via IK.
#
# Usage (all arguments are optional; defaults shown below):
#
#   bash bin/process_handumi_to_lerobot.sh \
#       --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
#       --embodiment piper
#
# Output defaults to NONHUMAN-RESEARCH/handumi-dataset-v2-piper under
# outputs/datasets/handumi-dataset-v2-piper.
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
    echo "[process_handumi_to_lerobot.sh] Virtual environment activated: ${VENV}"
else
    echo "[process_handumi_to_lerobot.sh] WARNING: no .venv found at ${VENV}. Using system Python."
fi

# ── Make the handumi package importable ────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ── Default arguments (override via CLI or env vars) ──────────────────────────
REPO_ID="${REPO_ID:-NONHUMAN-RESEARCH/handumi-dataset-v2}"
ROOT="${ROOT:-}"
REVISION="${REVISION:-main}"
EMBODIMENT="${EMBODIMENT:-piper}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-}"
EPISODES="${EPISODES:-}"
TASK="${TASK:-}"

# ── Parse CLI arguments ───────────────────────────────────────────────────────
EXTRA_FLAGS=()
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)          REPO_ID="$2";          shift 2 ;;
        --root)             ROOT="$2";             shift 2 ;;
        --revision)         REVISION="$2";         shift 2 ;;
        --embodiment)       EMBODIMENT="$2";       shift 2 ;;
        --output-repo-id)   OUTPUT_REPO_ID="$2";   shift 2 ;;
        --episodes)         EPISODES="$2";         shift 2 ;;
        --task)             TASK="$2";             shift 2 ;;
        --push-to-hub)
            EXTRA_FLAGS+=("$1"); shift ;;
        *)
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Print configuration ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   handumi - HandUMI to LeRobot embodiment conversion      ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Source repo   : %-40s║\n" "${REPO_ID}"
printf "║  Source root   : %-40s║\n" "${ROOT:-<derived from repo-id>}"
printf "║  Revision      : %-40s║\n" "${REVISION}"
printf "║  Embodiment    : %-40s║\n" "${EMBODIMENT}"
printf "║  Output repo   : %-40s║\n" "${OUTPUT_REPO_ID:-<derived>}"
printf "║  Episodes      : %-40s║\n" "${EPISODES:-all}"
printf "║  Task override : %-40s║\n" "${TASK:-none}"
printf "║  Extra flags   : %-40s║\n" "${EXTRA_FLAGS[*]:-none}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

CMD=(
    handumi-convert
    --repo-id "${REPO_ID}"
    --revision "${REVISION}"
    --embodiment "${EMBODIMENT}"
)

if [[ -n "${ROOT}" ]]; then
    CMD+=(--root "${ROOT}")
fi

if [[ -n "${OUTPUT_REPO_ID}" ]]; then
    CMD+=(--output-repo-id "${OUTPUT_REPO_ID}")
fi

if [[ -n "${EPISODES}" ]]; then
    CMD+=(--episodes "${EPISODES}")
fi

if [[ -n "${TASK}" ]]; then
    CMD+=(--task "${TASK}")
fi

CMD+=("${EXTRA_FLAGS[@]}" "${PASSTHROUGH[@]}")

exec "${CMD[@]}"
