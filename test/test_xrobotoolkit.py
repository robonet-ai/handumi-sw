"""
Test script for xrobotoolkit_sdk initialization and basic data readout.

The XRoboToolkit service must be launched before xrt.init() is called,
otherwise the C++ gRPC client crashes with:
  "terminate called without an active exception / Aborted (core dumped)"

This script mirrors the initialization sequence used in the production
pico_manager_thread_server.py to avoid that crash.
"""

import subprocess
import time

# ── 1. Launch the robotics service ───────────────────────────────────────────
SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"
SERVICE_WAIT_S = 3.0  # seconds to wait after launching before connecting

print(f"[xrt-test] Launching robotics service: {SERVICE_SCRIPT}")
try:
    subprocess.Popen(["bash", SERVICE_SCRIPT])
except FileNotFoundError:
    print(
        f"[xrt-test] WARNING: service script not found at {SERVICE_SCRIPT}.\n"
        "           Make sure the XRoboToolkit PC service is installed.\n"
        "           Continuing anyway in case the service is already running."
    )

print(f"[xrt-test] Waiting {SERVICE_WAIT_S}s for service to start …")
time.sleep(SERVICE_WAIT_S)

# ── 2. Import and initialize SDK ──────────────────────────────────────────────
try:
    import xrobotoolkit_sdk as xrt
except ImportError as exc:
    raise SystemExit(
        f"[xrt-test] ERROR: could not import xrobotoolkit_sdk: {exc}\n"
        "           Run  bin/install.sh  to build and install the SDK."
    ) from exc

print("[xrt-test] Calling xrt.init() …")
xrt.init()
print("[xrt-test] xrt.init() succeeded")

# ── 3. Wait for body-tracking data ───────────────────────────────────────────
BODY_POLL_S = 1.0
BODY_TIMEOUT_S = 15.0

print("[xrt-test] Waiting for body tracking data …")
deadline = time.time() + BODY_TIMEOUT_S
while not xrt.is_body_data_available():
    if time.time() > deadline:
        raise SystemExit(
            f"[xrt-test] ERROR: body data not available after {BODY_TIMEOUT_S}s.\n"
            "           Is the Pico headset connected and tracking active?"
        )
    print("  … still waiting")
    time.sleep(BODY_POLL_S)

print("[xrt-test] Body data is available!")

# ── 4. Read one frame and print a summary ─────────────────────────────────────
import numpy as np  # noqa: E402  (after SDK import so errors surface cleanly)

body_poses = xrt.get_body_joints_pose()
body_poses_np = np.array(body_poses)
stamp_ns = xrt.get_time_stamp_ns()

print(f"[xrt-test] Timestamp (ns): {stamp_ns}")
print(f"[xrt-test] Body poses shape: {body_poses_np.shape}")
print(f"[xrt-test] Root position (joint 0): {body_poses_np[0, :3]}")
print(f"[xrt-test] Root quaternion (joint 0, xyzw): {body_poses_np[0, 3:]}")
print("[xrt-test] PASS — xrobotoolkit_sdk is working correctly.")
