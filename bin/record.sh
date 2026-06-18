#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bin/record.sh  –  launcher for test/read_pico_cameras_motors.py
#
# Usage (all arguments are optional; defaults shown below):
#
#   bash bin/record.sh \
#       --cam-ids 0 2 4 \
#       --motor-port /dev/ttyUSB0 \
#       --motor-id leader \
#       --repo-id local/dexumi_dataset \
#       --output-dir datasets/my_dataset \
#       --task "Pick and place cube" \
#       --num-episodes 10 \
#       --episode-time-s 60 \
#       --fps 30 \
#       --vcodec h264
#
# Extra flags:
#   --push-to-hub          Upload to HuggingFace Hub after recording
#   --no-video             Save images as PNG instead of video
#   --skip-pico            Record without PICO headset (cameras + motors only)
#   --skip-adb-check       Don't wait for ADB device (useful if adb is absent)
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
    echo "[record.sh] Virtual environment activated: ${VENV}"
else
    echo "[record.sh] WARNING: no .venv found at ${VENV}. Using system Python."
fi

# ── Make the dexumi package importable ────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ── Default arguments (override via CLI) ──────────────────────────────────────
CAM_IDS="${CAM_IDS:-0 2 4}"          # space-separated camera indices
MOTOR_PORT="${MOTOR_PORT:-/dev/ttyUSB0}"
MOTOR_ID="${MOTOR_ID:-leader}"
REPO_ID="${REPO_ID:-local/dexumi_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/datasets/dexumi_dataset}"
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
        --motor-port)    MOTOR_PORT="$2";    shift 2 ;;
        --motor-id)      MOTOR_ID="$2";      shift 2 ;;
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
        --push-to-hub|--no-video|--skip-pico|--skip-adb-check)
            EXTRA_FLAGS+=("$1"); shift ;;
        *)
            # Forward unknown arguments directly to the Python script
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Print configuration ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          dexumi  –  multi-modal recording                ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Cameras       : %-40s║\n" "${CAM_IDS}"
printf "║  Motor port    : %-40s║\n" "${MOTOR_PORT}"
printf "║  Motor id      : %-40s║\n" "${MOTOR_ID}"
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
# shellcheck disable=SC2086
exec python "${REPO_ROOT}/test/read_pico_cameras_motors.py" \
    --cam-ids ${CAM_IDS} \
    --cam-width  "${CAM_WIDTH}" \
    --cam-height "${CAM_HEIGHT}" \
    --cam-fps    "${CAM_FPS}" \
    --motor-port "${MOTOR_PORT}" \
    --motor-id   "${MOTOR_ID}" \
    --repo-id    "${REPO_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --task       "${TASK}" \
    --num-episodes "${NUM_EPISODES}" \
    --episode-time-s "${EPISODE_TIME_S}" \
    --fps  "${FPS}" \
    --vcodec "${VCODEC}" \
    "${EXTRA_FLAGS[@]}" \
    "${PASSTHROUGH[@]}"
