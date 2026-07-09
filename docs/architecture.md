# HandUMI Software Architecture

Ultima modificacion: 2026-06-29

`handumi-sw` contiene el codigo real para captura raw, configuracion de
hardware, datasets, retargeting, IK y replay. La regla principal es simple:

```text
HandUMI raw data = source of truth
robot-specific datasets = derived artifacts
```

La captura no debe depender de Piper, Axol, Viser, pyroki ni nombres de joints
de robots. Los robots aparecen despues, en conversion offline, replay o
deployment.

## Data Flow

```text
HandUMI hardware
  wrist cameras + Feetech encoders + optional tracking
        |
        v
HandUMI raw LeRobot dataset
  robot-agnostic
        |
        v
Offline retargeting / IK
        |
        v
Robot-specific LeRobot dataset
  Piper / Axol / other embodiments
        |
        v
Replay, training, deployment
```

The implemented paths are:

```text
HandUMI raw record:
wrist cameras + Feetech widths -> raw LeRobot dataset

Optional tracking record:
wrist cameras + Feetech widths + PICO/Meta Quest poses -> raw LeRobot dataset

Robot-specific conversion:
PICO body pose -> retargeting -> IK -> Piper/Axol dataset -> replay/sim
```

## Project Layout

```text
.
|-- assets/                  # Robot URDFs and meshes
|-- bin/                     # Shell launchers
|-- configs/                 # Hardware/configuration defaults
|-- docs/                    # Architecture and embodiment guide
|-- scripts/                 # Manual hardware and pipeline scripts
|-- src/handumi/             # Core package
|-- tests/                   # Automated tests
`-- utils/                   # Upload/helper scripts
```

```text
src/handumi/
|-- capture/                 # Recording loop and capture features
|-- cameras/                 # USB cameras and preview helpers
|-- tracking/                # PICO / tracking backends
|-- feetech/                 # Feetech servo encoders and calibration
|-- dataset/                 # LeRobot schemas, readers, writers, conversion
|-- retargeting/             # Human/wearable/PICO poses -> robot targets
|-- robots/                  # Embodiment registry, IK specs, sim wiring
|-- replay/                  # PICO IK replay and robot hardware replay
`-- utils/
```

## Module Boundaries

| Module | Owns | Must not own |
|--------|------|--------------|
| `capture/` | episode timing, frame reads, raw recording | robot IK, robot joint names |
| `cameras/` | camera discovery/read/preview | dataset conversion |
| `tracking/` | PICO/tracker pose reads | robot-specific transforms |
| `feetech/` | servo IDs, encoder reads, width calibration | robot action layout |
| `dataset/` | LeRobot IO, schema, metadata, conversion | hardware polling |
| `retargeting/` | pose-to-target logic, axis maps | robot hardware replay |
| `robots/` | URDF names, IK specs, command layout, registry | capture logic |
| `replay/` | visualization/replay/deployment tools | raw recording |

## Raw Dataset Contract

The canonical HandUMI raw state is `float32[16]`:

```text
0   left_x
1   left_y
2   left_z
3   left_qx
4   left_qy
5   left_qz
6   left_qw
7   right_x
8   right_y
9   right_z
10  right_qx
11  right_qy
12  right_qz
13  right_qw
14  left_gripper_width
15  right_gripper_width
```

Gripper widths in the raw state are calibrated widths in meters. Feetech
auxiliary features also store raw ticks, normalized width, and width in mm for
hardware validation.

Core raw features:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.state
action
observation.feetech.left_ticks
observation.feetech.right_ticks
observation.feetech.left_width_mm
observation.feetech.right_width_mm
timestamp
frame_index
episode_index
task_index
```

`src/handumi/dataset/raw.py` is the code contract for the 16D layout. Use
`HANDUMI_RAW_STATE_NAMES` and `HANDUMI_RAW_STATE_SIZE` instead of repeating the
schema in scripts.

PICO/body/Feetech auxiliary signals can be additive features, but they should not
replace the compact raw state contract.

Feetech is used only as an encoder source for gripper aperture. It is not a
robot-arm dependency.

## Configs

```text
configs/cameras.yaml              # left_wrist/right_wrist assignment
configs/feetech.yaml              # left/right servo IDs and calibration ticks
configs/tracking_meta_quest.yaml  # Meta Quest backend settings
```

Initial Feetech convention:

```text
servo ID 0 -> left HandUMI gripper
servo ID 1 -> right HandUMI gripper
```

Setup is side-by-side, not blind global assignment:

```text
1. identify left/right serial ports by unplug/plug
2. scan Feetech IDs on those ports
3. assign ID 0 to left and ID 1 to right if needed
4. save left.port and right.port in configs/feetech.yaml
5. validate gripper encoder ticks while opening/closing
6. identify left/right USB cameras physically
7. record using configs/cameras.yaml
```

Each gripper can have its own USB serial port, or both can share one Feetech
bus. `configs/feetech.yaml` stores the left/right port mapping, closed/open
encoder ticks, and `max_width_mm`.

## Robot Embodiments

Robot-specific behavior is loaded through:

```python
from handumi.robots.registry import load_embodiment

runtime = load_embodiment("piper")
solver = runtime.solver_cls(config=runtime.config_cls())
sim = runtime.make_sim()
```

Each robot package contributes configuration, not algorithms:

```text
src/handumi/robots/<name>/
|-- shared.py       # URDF names, command layout, unit conversion
|-- solver.py       # RobotKinematicsSpec + KinematicsSolver binding
`-- retargeting.py  # RetargetingSpec binding
```

Shared robot logic lives once:

```text
robots/kinematics.py     # BimanualPyrokiSolver
robots/registry.py       # load_embodiment("piper" | "axol")
retargeting/pico_to_robot.py
retargeting/handumi_to_robot.py
```

Simulation/visualization is robot-agnostic and lives in its own package
(``robots/registry.py`` wires a specific embodiment's data into it, but
neither engine hard-codes any robot):

```text
sim/viser_sim.py         # ViserSim — kinematics-only web rendering
sim/mujoco_sim.py        # MujocoSim — headless real physics (contact/grasp)
assets/scenes/<name>/    # task scene assets (MJCF), e.g. cube_in_box
```

Per-arm command vectors are `(8,)`:

```text
Piper: [j1, j2, j3, j4, j5, j6, unused, gripper]
Axol : [j1, j2, j3, j4, j5, j6, j7, gripper]
```

## Phase 2 Tracking / Live Viser

After the cameras + Feetech checkpoint is validated on hardware, the next
tracking backend is Meta Quest through WebXR. The planned live path is:

```text
Meta Quest WebXR gripSpace poses
  + Feetech gripper widths
  -> HandUMI raw state
  -> retargeting / IK
  -> ViserSim live robot visualization
```

Details are tracked in [phase-2-motion-tracking.md](phase-2-motion-tracking.md).

## Manual Scripts

During hardware setup, HandUMI uses direct scripts from the repo instead of
installed CLIs. Once the hardware flow is stable, the validated commands can be
promoted to packaged entrypoints.

Setup scripts (interactive hardware setup, kept as direct scripts):

```text
scripts/setup/setup_ports.py
scripts/setup/calibrate_grippers.py
scripts/setup/home_servos.py
```

Capture entrypoints — run as modules (editable install, no PYTHONPATH):

```text
python -m handumi.capture.teleoperate_handumi     # live monitor, no saving
python -m handumi.capture.record_handumi_pico      # record: PICO tracking
python -m handumi.capture.record_handumi_quest     # record: Meta Quest tracking
python -m handumi.capture.live_tracking_quest       # Quest-only Rerun 3D viewer
```

The two tracking backends are deliberately separate: PICO
(`handumi.tracking.pico`, via XRoboToolkit) and Meta Quest
(`handumi.tracking.meta_quest`, via TCP/JSON + UDP sync). Each has its own
recorder; the user picks the one matching the hardware they have. Both emit the
same 16D raw state, so downstream (dataset, conversion) is backend-agnostic.

Legacy robot/retarget scripts (offline dataset -> robot embodiment; kept for the
Piper/arm path, not part of capture):

```text
scripts/process_handumi_to_lerobot.py  -> handumi.dataset.conversion
handumi-replay-in-sim                  -> handumi.scripts.replay.replay_in_sim
```

`python -m handumi.capture.teleoperate_handumi` is the LeRobot-style live
inspection loop. It does not write a dataset; it streams cameras and Feetech
aperture signals to Rerun so the operator can validate hardware before
recording.

Shell launchers:

```text
bin/record_pico.sh                     # PICO recorder launcher (capture)
bin/process_handumi_to_lerobot.sh      # legacy: dataset -> robot embodiment
bin/piper/replay_from_dataset.sh       # legacy: replay a dataset on Piper
```

Automated tests live only under `tests/`.

## Invariants

- Raw recording remains robot-agnostic.
- Robot datasets are reproducible from raw/source data plus config.
- Scripts stay thin; reusable logic lives in `src/handumi`.
- New robots are added through `robots/<name>/` and the registry.
