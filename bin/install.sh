#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv sync
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

ensure_xrobotoolkit_sources
ensure_xrobotoolkit_native_lib
ensure_venv
ensure_project_deps
ensure_xrobotoolkit_python_package

echo "==> Installation complete"
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo ""

# ── Runtime note: XRoboToolkit service ───────────────────────────────────────
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
