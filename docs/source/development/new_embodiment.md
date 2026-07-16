# Add a New Robot Embodiment

Ultima modificacion: 2026-07-15 11:26:49 -05 -0500

The HandUMI recorder and raw dataset are robot agnostic. An embodiment is a
downstream adapter for conversion, simulation, or real-robot control; adding
one must not add vendor-specific requirements to HandUMI setup or recording.

Current supported scope: fixed-base bimanual robots with one TCP and one
parallel gripper per side.

Adding a model for replay/simulation is configuration-driven. Adding real
hardware also requires a hardware backend; real teleoperation is currently
implemented only for Piper.

## Required Files

```text
assets/<robot>/
├── <robot>.urdf
├── <robot>.xml              # optional; required for MuJoCo contact physics
└── meshes/

configs/robots/
└── <robot>.yaml

configs/calibration/
├── <robot>_<device>_controller_tcp.yaml
└── <robot>_table.yaml       # required for absolute-table replay
```

## 1. Add the Robot Model

Place the URDF and meshes under `assets/<robot>/`.

The current loader requires:

- both arms in one URDF;
- actuated joints prefixed with `left_` and `right_`;
- one TCP link per side at the physical gripper tip;
- a fixed base;
- one parallel gripper per side.

These are current implementation constraints, not requirements of the raw
HandUMI dataset format.

Use `package://...` mesh paths when needed. `pkg_root` in the robot YAML
resolves them.

## 2. Add the Robot Configuration

Create `configs/robots/<robot>.yaml`:

```yaml
kind: myrobot
urdf: assets/myrobot/myrobot.urdf
mjcf: assets/myrobot/myrobot.xml
pkg_root: assets/myrobot

ee_links:
  left: left_tcp
  right: right_tcp

home_q: [0.0, 0.0]
gripper_max_width_m: 0.07

# Physical gripper and mount installed on HandUMI for this robot.
handumi_tool:
  gripper: myrobot_parallel_v1
  controller_mount: handumi_v1

# Add an entry only after calibrating this exact robot/gripper/controller setup.
controller_tcp_calibrations:
  meta: configs/calibration/myrobot_meta_controller_tcp.yaml

ik_weights:
  pos: 100.0
  ori: 15.0
  rest: 2.0
  max_joint_delta: 0.06981317
  max_reach: 0.45
```

`home_q` must contain one value for every actuated URDF joint, including any
actuated gripper joints.

Existing references:

- [Piper configuration](https://github.com/robonet-ai/handumi-sw/blob/main/configs/robots/piper.yaml)
- [Axol configuration](https://github.com/robonet-ai/handumi-sw/blob/main/configs/robots/axol.yaml)

## 3. Discover the Model

Robot names are discovered from `configs/robots/*.yaml`; no central list needs
editing. A real robot additionally registers one lazy backend factory so its
vendor SDK is imported only when selected.

If the robot assets must be shipped in the wheel, add them to the package-data
configuration in `pyproject.toml`.

## 4. Calibrate the HandUMI TCP for This Robot

Controller-to-TCP calibration belongs to the complete physical assembly:

```text
robot + gripper/tool + HandUMI mount + tracking controller + side
```

Do not reuse Piper calibration for another robot or gripper.

Mount the new robot's gripper/tool on HandUMI and perform the pivot procedure
for both sides. Follow
[HandUMI Setup and Calibration](../setup.md).

Write the final calibration to:

```text
configs/calibration/<robot>_<device>_controller_tcp.yaml
```

Then reference that file from the robot YAML:

```yaml
handumi_tool:
  gripper: myrobot_parallel_v1
  controller_mount: handumi_v1

controller_tcp_calibrations:
  meta: configs/calibration/myrobot_meta_controller_tcp.yaml
```

Changing the gripper, printed mount, controller mount, or mechanical TCP
requires a new calibration and a new gripper/mount identifier.

## 5. Calibrate Robot Placement Relative to the Table

Absolute-table replay also needs `robot_from_table`:

```yaml
verified: true
calibration:
  robot_from_table:
    position: [0.0, 0.0, 0.0]
    quaternion: [0.0, 0.0, 0.0, 1.0]
```

Save it as:

```text
configs/calibration/<robot>_table.yaml
```

Controller-to-TCP and `robot_from_table` solve different problems:

- Controller-to-TCP reconstructs the demonstrated gripper-tip trajectory.
- `robot_from_table` places that trajectory in the robot workspace.

Do not compensate an incorrect TCP offset by changing `robot_from_table`.

## 6. Record a Validation Dataset

Record using the intended embodiment:

```bash
uv run handumi-record \
  --device meta \
  --robot myrobot \
  --repo-id local/myrobot_validation \
  --output-dir outputs/myrobot_validation \
  --task "myrobot TCP validation" \
  --num-episodes 1 \
  --no-sounds
```

Recording snapshots the robot, gripper, controller mount, calibration hash,
and both Controller-to-TCP transforms into dataset metadata. Raw controller
poses remain unchanged.

## 7. Replay and Inspect Geometry

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --repo-id local/myrobot_validation \
  --dataset-root outputs/myrobot_validation \
  --episode 0 \
  --robot myrobot \
  --retarget-mode absolute-table
```

Before IK, replay reports:

- source robot, gripper, controller, and mount;
- selected calibration and SHA-256;
- Controller-to-TCP distance for each side;
- minimum calibrated TCP height;
- minimum same-frame distance between both TCP trajectories;
- selected `robot_from_table` translation.

Calibration precedence is:

1. explicit `--controller-tcp-calibration` override;
2. an identity-bound calibration snapshot in the dataset;
3. the source robot/device calibration configured in its robot YAML;
4. legacy device fallback when no robot/tool identity exists.

Legacy snapshots can be forced only for investigation with
`--use-dataset-tcp-calibration`.

## 8. Verify Simulation and IK

```bash
uv run pytest -q \
  tests/retargeting/test_handumi_to_robot.py \
  tests/scripts/test_replay_in_sim.py
```

Check the replay output for:

- no missing mesh, joint, or TCP link;
- initial TCP position error within tolerance;
- acceptable per-frame IK position and rotation error;
- the expected table contact height;
- both arms reaching the intended shared task region.

Large IK errors commonly indicate an incorrect TCP link, unreachable
`robot_from_table`, invalid `home_q`, or unsuitable IK weights.

## 9. Add Real Hardware Support

Model replay does not automatically provide real hardware control.

A new real robot implements the shared backend contract: prepare transport,
connect, read/hold feedback, home, command joints/grippers, report health, and
close safely. Keep vendor units, SDK objects, communication threads, and
watchdog behavior inside that backend. The shared teleoperation engine and
dataset code continue using meters, radians, normalized gripper openings, and
pose7 quaternions in XYZW order.

## Completion Checklist

- [ ] Combined bimanual URDF loads.
- [ ] Left and right TCP links are at the physical gripper tips.
- [ ] `home_q` matches the actuated-joint count.
- [ ] Robot name is registered.
- [ ] Assets are included in packaging.
- [ ] HandUMI gripper and controller mount are identified.
- [ ] Left and right pivot calibrations pass residual checks.
- [ ] Robot/device calibration is referenced from the robot YAML.
- [ ] `robot_from_table` is measured and marked verified.
- [ ] Validation recording contains identity-bound calibration metadata.
- [ ] Replay diagnostics and IK errors are acceptable.
- [ ] Real hardware backend exists if physical teleoperation is required.
