# Add a New Robot Embodiment

HandUMI recordings are robot agnostic. An *embodiment* adapts those recordings
to a robot for simulation, replay, or real teleoperation.

This guide covers fixed-base bimanual robots with one TCP and one parallel
gripper per side. Start by choosing the scope of the contribution:

| Scope | Required |
| --- | --- |
| Simulation / replay | URDF, assets, robot YAML, tests, and replay evidence. |
| Absolute-table replay | Above, plus a `<robot>_table.yaml` placement. |
| Real hardware | Above, plus a backend, physical calibrations, and hardware safety tests. |

Do not add vendor SDKs, local ports, credentials, or robot-specific recording
requirements to the HandUMI recorder.

## 1. Prepare source assets

Create this layout:

```text
assets/<robot>/
├── README.md                 # vendor URL, commit/release, license, derivations
├── LICENSE.<vendor>          # when redistribution requires it
├── <robot>.urdf
├── <robot>.xml               # optional MuJoCo contact model
└── meshes/

configs/robots/
└── <robot>.yaml
```

Use official assets when possible. In `assets/<robot>/README.md`, record the
source repository, immutable commit or release, license, and every generated
mesh. Do not commit unlicensed assets, local calibration files, datasets, or
`configs/rig.yaml`.

If the vendor supplies two single-arm URDFs, build one combined bimanual URDF:

- namespace every copied link and joint as `left_` or `right_`;
- preserve the vendor joint origins, axes, limits, and mesh transforms;
- add fixed mounts from one shared base to each arm;
- add one fixed TCP link per side at the actual grasp point;
- keep gripper joints actuated and give all visual mesh paths a resolvable
  `package://` or `pkg_root` path.

The robot name and joint names are declared in YAML, so no central registry is
needed for simulation. Add `assets/<robot>` to the wheel force-include section
of `pyproject.toml` when the assets must ship with the package.

## 2. Add the robot YAML

Create `configs/robots/<robot>.yaml`:

```yaml
kind: myrobot
urdf: assets/myrobot/myrobot.urdf
pkg_root: assets/myrobot
# mjcf: assets/myrobot/myrobot.xml  # only if a MuJoCo model exists

arms:
  left:
    ee_link: left_tcp
    joint_names: [left_joint1, left_joint2]
    gripper_joints:
      - {name: left_finger_joint, closed: 0.0, open: 0.035}
  right:
    ee_link: right_tcp
    joint_names: [right_joint1, right_joint2]
    gripper_joints:
      - {name: right_finger_joint, closed: 0.0, open: 0.035}

# One value for every actuated URDF joint, including fingers.
home_q: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
gripper_max_width_m: 0.07

ik_weights:
  pos: 100.0
  ori: 15.0
  rest: 2.0
  max_joint_delta: 0.06981317
  max_reach: 0.45

replay:
  max_joint_delta: 0.20
  # Use physical-width when source and target grippers have different strokes.
  # gripper_retarget: physical-width
```

`closed` and `open` are the actual URDF joint values for normalized HandUMI
opening `0` and `1`. Verify them visually; URDF joint signs are not universal.
Use `gripper_retarget: physical-width` to preserve aperture in meters when the
target gripper has a different maximum width. The default, `normalized`,
preserves the source opening percentage.

Useful references are `configs/robots/piper.yaml`, `openarmv1.yaml`,
`trlc_dk1.yaml`, and `yam.yaml`.

## 3. Add a simulation test

Add a focused test in `tests/robots/test_registry.py` that checks:

- left/right arm joint order and indices;
- home pose length and a symmetric FK sanity check;
- gripper closed/open mapping, if present;
- `load_urdf(load_meshes=True)` resolves every visual mesh.

Run it before attempting replay:

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/robots/test_registry.py
JAX_PLATFORMS=cpu uv run python -c \
  "from handumi.robots.registry import load_embodiment; print(load_embodiment('myrobot').joint_names)"
```

## 4. Calibrate only when the scope needs it

### Table placement for absolute replay

Add `configs/calibration/<robot>_table.yaml`. This is the transform from the
demonstration table frame to the target robot world; it is not a TCP offset.

```yaml
schema_version: 1
kind: handumi_robot_table_calibration
robot: myrobot
source: measured_installation
verified: false
calibration:
  frame_convention: pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion
  robot_from_table:
    position: [0.0, 0.0, 0.0]
    quaternion: [0.0, 0.0, 0.0, 1.0]
```

Use `verified: false` for a simulation placement. Set it to `true` only after
measuring the physical installation. Never compensate a wrong TCP calibration
by changing this transform.

### HandUMI controller-to-TCP calibration

This calibration belongs to the physical HandUMI assembly:

```text
tracking controller + HandUMI mount + HandUMI gripper/tool + side
```

It does **not** automatically change because the target robot changes. A
simulation-only PR needs no new pivot capture. Reuse an identity-bound dataset
snapshot only when the physical HandUMI assembly is exactly the same; otherwise
follow [HandUMI Setup and Calibration](../setup.md) and reference the new file
from `controller_tcp_calibrations` and `handumi_tool` in the robot YAML.

## 5. Replay a validation episode

For absolute-table replay, use a recorded validation episode and inspect the
model in Viser:

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --repo-id local/myrobot_validation \
  --dataset-root outputs/myrobot_validation \
  --episode 0 --robot myrobot --retarget-mode absolute-table --strict-ik
```

Check mesh loading, home pose, TCP placement, gripper direction and aperture,
table height, shared workspace, and reported IK errors. A large error usually
means a bad TCP, placement, home pose, joint order, or IK limit.

## 6. Add real hardware support only when ready

Replay support does not provide robot control. A hardware PR must implement
the `RobotBackend` contract in `src/handumi/real/backends/__init__.py`:
prepare, connect, home, command, hold, health check, and close. Register it
lazily in `make_real_backend`, declare it in the robot YAML as
`real.backend`, keep vendor units and SDK code inside that backend, and add
backend tests. `configs/rig.yaml` should only hold local transport details
such as CAN ports. Real control uses radians, meters, normalized openings, and
XYZW pose quaternions.

## 7. Open the pull request

Before opening the PR, run:

```bash
JAX_PLATFORMS=cpu uv run pytest -q
uv build
.venv-docs/bin/sphinx-build -W -b html docs/source /tmp/handumi-docs
git status --short
```

The PR description should state:

- robot model, vendor source, immutable revision, and license;
- scope: simulation, absolute replay, and/or real hardware;
- frame convention, TCP and gripper mapping decisions;
- replay command, frame count, and position/rotation error summary;
- a simulator screenshot or short recording;
- physical calibration evidence only when claiming hardware support.

## Completion checklist

- [ ] Assets and their provenance/license are committed.
- [ ] Combined URDF and all meshes load.
- [ ] TCPs, joint order, limits, home pose, and gripper mapping are tested.
- [ ] Robot assets are included in the wheel.
- [ ] Table calibration is present for absolute-table replay.
- [ ] `verified: true` is used only for a measured physical installation.
- [ ] Replay passes with acceptable IK and gripper aperture.
- [ ] Real backend and safety tests exist if real teleoperation is claimed.
- [ ] README/docs are updated and the full test, build, and docs checks pass.
