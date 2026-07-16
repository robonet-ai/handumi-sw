#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_XRT=0
ROBOT=""
WITH_SIM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-xrt) SKIP_XRT=1 ;;
    --sim) WITH_SIM=1 ;;
    --robot)
      shift
      if [[ $# -eq 0 || ( "$1" != "piper" && "$1" != "openarmv1" ) ]]; then
        echo "error: --robot expects piper or openarmv1" >&2
        exit 1
      fi
      ROBOT="$1"
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "       usage: $0 [--skip-xrt] [--sim] [--robot piper|openarmv1]" >&2
      exit 1
      ;;
  esac
  shift
done

XROBO_DIR="external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
SIBLING_XROBO="$ROOT/../GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
XROBO_REPO="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git"
ARCH="$(uname -m)"
VENV_DIR=".venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

activate_venv() {
  export VIRTUAL_ENV="$ROOT/$VENV_DIR"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  unset PYTHONHOME
}

pip_installed() {
  uv pip show "$1" >/dev/null 2>&1
}

ensure_rig_config() {
  if [[ -f "configs/rig.yaml" ]]; then
    echo "==> Reusing local rig configuration (configs/rig.yaml)"
    return 0
  fi
  cp configs/rig.example.yaml configs/rig.yaml
  echo "==> Created configs/rig.yaml from the example; edit it for this machine"
}

# ── XRoboToolkit sources ──────────────────────────────────────────────────────

ensure_xrobotoolkit_sources() {
  if [[ -d "$XROBO_DIR" ]]; then
    echo "==> XRoboToolkit sources already present, skipping fetch"
    return 0
  fi

  mkdir -p external_dependencies

  if [[ -d "$SIBLING_XROBO" ]]; then
    echo "==> Copying XRoboToolkit dependency from GR00T-WholeBodyControl"
    cp -a "$SIBLING_XROBO" "$XROBO_DIR"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "error: missing dependency directory: $XROBO_DIR" >&2
    echo "       Install git or copy the dependency manually into external_dependencies/." >&2
    exit 1
  fi

  echo "==> Cloning XRoboToolkit-PC-Service-Pybind"
  git clone "$XROBO_REPO" "$XROBO_DIR"
}

ensure_xrobotoolkit_native_lib() {
  if [[ "$ARCH" == "x86_64" ]] && [[ -f "$XROBO_DIR/lib/libPXREARobotSDK.so" ]]; then
    echo "==> PXREARobotSDK x86_64 library already built, skipping"
    return 0
  fi

  if [[ "$ARCH" == "aarch64" ]] && [[ -f "$XROBO_DIR/lib/aarch64/libPXREARobotSDK.so" ]]; then
    echo "==> PXREARobotSDK aarch64 library already built, skipping"
    return 0
  fi

  if [[ "$ARCH" == "x86_64" ]]; then
    echo "==> Building PXREARobotSDK for x86_64"
    XRT_TMP="$XROBO_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [[ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]]; then
      git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XROBO_DIR/lib" "$XROBO_DIR/include"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
      "$XROBO_DIR/include/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
      "$XROBO_DIR/include/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
      "$XROBO_DIR/lib/"
    rm -rf "$XRT_TMP"
    echo "==> PXREARobotSDK x86_64 native library built and installed"
    return 0
  fi

  if [[ "$ARCH" == "aarch64" ]]; then
    echo "==> Building PXREARobotSDK for aarch64"
    XRT_TMP="$XROBO_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [[ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]]; then
      git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XROBO_DIR/lib/aarch64" "$XROBO_DIR/include/aarch64"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
      "$XROBO_DIR/include/aarch64/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
      "$XROBO_DIR/include/aarch64/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
      "$XROBO_DIR/lib/aarch64/"
    rm -rf "$XRT_TMP"
    echo "==> PXREARobotSDK aarch64 native library built and installed"
    return 0
  fi

  echo "error: unsupported architecture for XRoboToolkit: $ARCH" >&2
  exit 1
}

# ── Venv + project deps ───────────────────────────────────────────────────────

ensure_venv() {
  if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    echo "==> Reusing existing virtual environment ($VENV_DIR)"
  else
    echo "==> Creating uv virtual environment (Python >= 3.12)"
    uv venv --python ">=3.12"
  fi
  # Activate for all subsequent uv pip commands in this session.
  activate_venv
}

ensure_project_deps() {
  echo "==> Syncing project dependencies (update only if needed)"
  local extras=()
  [[ "$WITH_SIM" -eq 1 ]] && extras+=(--extra sim)
  [[ -n "$ROBOT" ]] && extras+=(--extra "${ROBOT/openarmv1/openarm}")
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv sync "${extras[@]}"
}

ensure_openarm_system_deps() {
  [[ "$ROBOT" == "openarmv1" ]] || return 0
  if command -v openarm-can-cli >/dev/null 2>&1 && \
     dpkg-query -W -f='${Status}' libopenarm-can-dev 2>/dev/null | grep -q "install ok installed"; then
    echo "==> OpenArm CAN system library and tools already installed"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: automatic OpenArm system setup currently supports Ubuntu/Debian" >&2
    exit 1
  fi
  echo "==> Installing official OpenArm CAN system packages"
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:openarm/main
  sudo apt-get update
  sudo apt-get install -y libopenarm-can-dev openarm-can-utils
}

# ── XRoboToolkit Python package ───────────────────────────────────────────────

ensure_xrobotoolkit_python_package() {
  if pip_installed xrobotoolkit-sdk || pip_installed xrobotoolkit_sdk; then
    echo "==> xrobotoolkit_sdk already installed, skipping rebuild"
    return 0
  fi

  echo "==> Installing XRoboToolkit SDK (editable, no build isolation)"
  # cmake and pybind11 are declared in pyproject.toml so uv sync already put
  # them in the venv before we reach this point.
  export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
  uv pip install --no-build-isolation -e "$XROBO_DIR/"
}

# ── Main ──────────────────────────────────────────────────────────────────────

ensure_rig_config
if [[ "$SKIP_XRT" -eq 1 ]]; then
  echo "==> --skip-xrt: skipping XRoboToolkit (PICO) sources/build/package"
else
  ensure_xrobotoolkit_sources
  ensure_xrobotoolkit_native_lib
fi
ensure_venv
ensure_openarm_system_deps
ensure_project_deps
if [[ "$SKIP_XRT" -ne 1 ]]; then
  ensure_xrobotoolkit_python_package
fi

echo "==> Installation complete"
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo ""

# ── Runtime note: XRoboToolkit service ───────────────────────────────────────
if [[ "$SKIP_XRT" -eq 1 ]]; then
  echo "NOTE: XRoboToolkit (PICO) was skipped (--skip-xrt). Using Meta Quest"
  echo "      tracking needs no PC service — see docs/README_quest.md."
else
  SERVICE_SCRIPT="/opt/apps/roboticsservice/runService.sh"
  if [[ -f "$SERVICE_SCRIPT" ]]; then
    echo "NOTE: XRoboToolkit PC service found at $SERVICE_SCRIPT"
    echo "      Start it before running any xrobotoolkit_sdk scripts:"
    echo "        bash $SERVICE_SCRIPT"
  else
    echo "WARNING: XRoboToolkit PC service not found at $SERVICE_SCRIPT"
    echo "         xrt.init() will crash (core dump) if the service is not running."
    echo "         Install the XRoboToolkit PC service from:"
    echo "           https://github.com/XR-Robotics/XRoboToolkit-PC-Service"
  fi
fi
