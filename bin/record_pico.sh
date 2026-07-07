#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bin/record_pico.sh  –  launcher for handumi.scripts.record_handumi_pico
#
# Usage (all arguments are optional; defaults shown below):
#
#   bash bin/record_pico.sh \
#       --repo-id local/handumi_dataset \
#       --output-dir datasets/my_dataset \
#       --task "Pick and place cube" \
#       --num-episodes 1 \
#       --episode-time-s 60 \
#       --fps 30 \
#       --vcodec h264
#
# Extra flags:
#   --push-to-hub          Upload to HuggingFace Hub after recording
#   --no-video             Save images as PNG instead of video
#   --use-pico             Enable PICO tracking streams
#   --skip-feetech         Record without Feetech gripper encoders
#   --skip-adb-check       Don't wait for ADB device (useful if adb is absent)
#   --laptop-camera        Add laptop camera with stopwatch + reach overlay
#   --no-laptop-preview    Do not open the live saved-video preview window
#   --manual-control       A=start/stop, B=repeat, Y=finish (PICO buttons)
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
    echo "[record_pico.sh] Virtual environment activated: ${VENV}"
else
    echo "[record_pico.sh] WARNING: no .venv found at ${VENV}. Using system Python."
fi

# ── Make the handumi package importable ────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ── Default arguments (override via CLI) ──────────────────────────────────────
CAM_IDS="${CAM_IDS:-}"              # Optional left_wrist right_wrist camera override
CAMERA_CONFIG="${CAMERA_CONFIG:-${REPO_ROOT}/configs/cameras.yaml}"
FEETECH_CONFIG="${FEETECH_CONFIG:-}"   # empty -> recorder resolves the per-user cache
FEETECH_PORT="${FEETECH_PORT:-}"
REPO_ID="${REPO_ID:-local/handumi_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/datasets/handumi_dataset}"
TASK="${TASK:-Teleoperation recording}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
FPS="${FPS:-30}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FPS="${CAM_FPS:-30}"
VCODEC="${VCODEC:-h264}"

# ── Parse extra flags from CLI arguments ──────────────────────────────────────
EXTRA_FLAGS=()
PASSTHROUGH=()  # arguments forwarded verbatim to the Python script

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cam-ids)
            shift
            CAM_IDS=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CAM_IDS="${CAM_IDS} $1"
                shift
            done
            CAM_IDS="${CAM_IDS# }"  # trim leading space
            ;;
        --feetech-config) FEETECH_CONFIG="$2"; shift 2 ;;
        --camera-config) CAMERA_CONFIG="$2"; shift 2 ;;
        --feetech-port)   FEETECH_PORT="$2";   shift 2 ;;
        --repo-id)       REPO_ID="$2";       shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --task)          TASK="$2";          shift 2 ;;
        --num-episodes)  NUM_EPISODES="$2";  shift 2 ;;
        --episode-time-s) EPISODE_TIME_S="$2"; shift 2 ;;
        --fps)           FPS="$2";           shift 2 ;;
        --cam-width)     CAM_WIDTH="$2";     shift 2 ;;
        --cam-height)    CAM_HEIGHT="$2";    shift 2 ;;
        --cam-fps)       CAM_FPS="$2";       shift 2 ;;
        --vcodec)        VCODEC="$2";        shift 2 ;;
        --push-to-hub|--no-video|--skip-pico|--use-pico|--skip-feetech|--skip-adb-check|--laptop-camera|\
        --manual-control|--no-laptop-overlay|--no-laptop-preview|--save-unreachable|--pico-mandos|\
        --pico-object|--pico-whole-body|--pico-adb|--pico-wifi)
            EXTRA_FLAGS+=("$1"); shift ;;
        *)
            # Forward unknown arguments directly to the Python script
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Print configuration ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                 handumi – PICO recording                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Cameras       : %-40s║\n" "${CAM_IDS:-from config}"
printf "║  Camera config : %-40s║\n" "${CAMERA_CONFIG}"
printf "║  Feetech config: %-40s║\n" "${FEETECH_CONFIG:-per-user cache}"
printf "║  Feetech port  : %-40s║\n" "${FEETECH_PORT:-from config}"
printf "║  Repo id       : %-40s║\n" "${REPO_ID}"
printf "║  Output dir    : %-40s║\n" "${OUTPUT_DIR}"
printf "║  Task          : %-40s║\n" "${TASK}"
printf "║  Episodes      : %-40s║\n" "${NUM_EPISODES}"
printf "║  Episode time  : %-40s║\n" "${EPISODE_TIME_S}s"
printf "║  FPS           : %-40s║\n" "${FPS}"
printf "║  Codec         : %-40s║\n" "${VCODEC}"
printf "║  Extra flags   : %-40s║\n" "${EXTRA_FLAGS[*]:-none}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Run the recorder ──────────────────────────────────────────────────────────
FEETECH_ARGS=()
if [[ -n "${FEETECH_CONFIG}" ]]; then
    FEETECH_ARGS+=(--feetech-config "${FEETECH_CONFIG}")
fi
if [[ -n "${FEETECH_PORT}" ]]; then
    FEETECH_ARGS+=(--feetech-port "${FEETECH_PORT}")
fi
CAMERA_ARGS=(--camera-config "${CAMERA_CONFIG}")
if [[ -n "${CAM_IDS}" ]]; then
    # shellcheck disable=SC2206
    CAMERA_ARGS+=(--cam-ids ${CAM_IDS})
fi

exec python -m handumi.scripts.record_handumi_pico \
    "${CAMERA_ARGS[@]}" \
    --cam-width  "${CAM_WIDTH}" \
    --cam-height "${CAM_HEIGHT}" \
    --cam-fps    "${CAM_FPS}" \
    "${FEETECH_ARGS[@]}" \
    --repo-id    "${REPO_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --task       "${TASK}" \
    --num-episodes "${NUM_EPISODES}" \
    --episode-time-s "${EPISODE_TIME_S}" \
    --fps  "${FPS}" \
    --vcodec "${VCODEC}" \
    "${EXTRA_FLAGS[@]}" \
    "${PASSTHROUGH[@]}"
