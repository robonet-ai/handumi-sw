# Robot-Agnostic Architecture Plan

Status: design and migration plan. The first implementation target remains
fixed-base bimanual manipulation with one TCP and one parallel gripper per arm.

Implemented baseline:

- robot/gripper/controller-mount identity is declared in robot configuration;
- new recordings snapshot identity-bound Controller-to-TCP metadata;
- replay and conversion prefer trusted source snapshots, then the source robot
  setup, while legacy datasets use the validated setup when available;
- real/sim teleoperation use the same robot/device calibration selection;
- replay reports calibration SHA-256 and source-geometry diagnostics.

The remaining phases below describe broader decoupling that is not yet
implemented.

## 1. Decision

HandUMI capture is already mostly independent of the target robot. It records
controller poses, gripper state, cameras, tracking health, timing, and spatial
calibration before robot IK. The main coupling is downstream:

- robot discovery and packaging;
- assumptions about `left_*` and `right_*` joint names;
- the combined bimanual URDF requirement;
- Piper imports and units in real teleoperation;
- duplicated real/simulation control state machines;
- robot-specific simulator fixes in generic code;
- Piper-only setup and deployment configuration.

Robot-agnostic means that core capture, retargeting, teleoperation, safety, and
simulation orchestration do not import vendor code, branch on robot names, or
infer semantics from joint-name prefixes. A new robot may require a model
profile and a hardware plugin, but it must not require edits to core modules.

This refactor must preserve current Piper behavior and replay of existing
datasets. It is a staged migration, not a flag-day rewrite.

### Initial scope

- Two task roles: `left` and `right`.
- One fixed robot base per arm or one combined fixed-base bimanual model.
- One TCP target and one parallel gripper command per role.
- Position-controlled arm joints.
- Viser visualization and MuJoCo simulation.
- Real-time teleoperation and offline replay/conversion.

The roles describe task semantics only. They do not require joints or links to
contain the words `left` or `right`.

### Non-goals for the first migration

- Mobile-base planning.
- Legged or whole-body control.
- Dexterous hands with many independently controlled fingers.
- Torque or impedance control as the common API.
- A universal vendor protocol.
- Runtime composition of arbitrary robot graphs.

The interfaces may expose capabilities for future extensions, but these cases
must not complicate the first stable contract.

## 2. Current Coupling Audit

| Boundary | Current coupling | Target owner |
| --- | --- | --- |
| Raw recording | `--robot` defaults to Piper and snapshots a target robot | Optional deployment intent; capture stays usable without a robot profile |
| Robot discovery | Static `EMBODIMENT_NAMES` | Dynamic profile discovery |
| Kinematic model | One combined URDF, fixed left/right prefixes and two fixed EEs | Explicit role, joint-group, base-link, and TCP declarations |
| IK | Prefix-derived indices, frame-based joint delta, unused tuning fields | Kinematics backend plus time-aware safety limits |
| Real teleoperation | Direct `PiperCanEnvironment` import and Piper unit conversion | Generic driver and shared controller |
| Simulation teleoperation | Control state machine embedded in the CLI | Same controller used by real and simulated backends |
| MuJoCo | Piper-oriented collision mutation and naming assumptions | Profile-owned actuator and collision policy |
| Hardware setup | Piper-only CAN flow | Plugin-owned setup command or documented setup hook |
| Packaging | Axol and Piper assets explicitly enumerated | Package-resource discovery and installed-wheel test |
| Calibration | P0 complete: identity-bound snapshots plus robot/device setup fallback | Preserve this contract while profiles become independently discoverable |

## 3. Target Architecture

```text
CaptureDevice + CaptureProfile
             |
             v
       HandUMI task-space sample/episode
             |
             v
          Retargeter <--------- DeploymentProfile
             |                         |
             v                         +--> RobotModelProfile
     task-role TCP targets             +--> DriverProfile
             |                         +--> RobotToolCalibration
             |                         +--> spatial/calibration references
             v
      KinematicsBackend
             |
             v
         SafetyFilter
             |
             v
        RobotCommand
          /       \
         v         v
   RobotDriver  SimulationBackend
```

### Configuration boundaries

#### Capture profile

Owns properties of the data-collection setup:

- tracking device and coordinate conventions;
- cameras and controller/gripper input mapping;
- capture-time tracking/world calibration;
- provenance and hashes for the calibration files used.

#### Robot model profile

Owns only embodiment and kinematic information:

- URDF or other model resource;
- canonical joint order and explicit joint groups;
- role-to-group, base-link, and TCP-link mapping;
- home posture and joint limits;
- gripper kinematics and command range;
- IK backend and robot-specific solver tuning;
- optional simulation resources and mappings.

Loading a model must not connect to hardware or import a vendor SDK.

#### Robot-tool calibration profile

Owns the Controller-to-TCP transform for the exact physical assembly used to
demonstrate a task:

- robot embodiment, such as Piper;
- gripper or tool variant mounted on HandUMI;
- tracking controller/mount variant, such as Meta left or right;
- side-specific `controller_from_tcp` or `tcp_from_controller` transforms;
- calibration method, sample count, residuals, date, and file hash.

This calibration is robot-specific in practice because changing from Piper to
another arm normally changes the gripper mounted on HandUMI and therefore the
Controller-to-TCP geometry. It is more precisely keyed by the tuple
`(robot, gripper/tool, controller mount, side)`, rather than by the robot name
alone.

The robot model profile declares its supported/default robot-tool calibration
profiles. Recording snapshots the selected calibration into the dataset, so
replay uses the geometry of the tool that actually produced the demonstration.
A later replay to another target robot must not replace that source calibration
with the new target robot's calibration.

#### Driver profile

Selects a hardware plugin and maps its joints and grippers to the canonical
model contract. It may contain non-secret transport defaults and capabilities.
Machine-local ports, CAN interfaces, device IDs, and overrides stay in
`rig.yaml` or an equivalent local configuration.

#### Deployment profile

Binds components that are valid together on a particular rig:

- robot model profile;
- driver profile or simulation backend;
- capture profile;
- robot-tool calibration profile;
- `robot_from_table` or equivalent spatial calibration;
- verified calibration status and provenance;
- optional safety-limit overrides that are stricter than the model limits.

The current Piper-to-Meta calibration association in
`configs/robots/piper.yaml` is conceptually correct. It should migrate to a
versioned robot-tool calibration reference, not be generalized into a single
Meta calibration shared by every robot.

## 4. Stable Runtime Contracts

All core values use SI units:

- joint angles in radians;
- positions and gripper widths in meters;
- velocities in radians/second or meters/second;
- timestamps in monotonic seconds;
- transforms named as `target_from_source`.

### State and command

```python
@dataclass(frozen=True)
class RobotState:
    timestamp_s: float
    joint_positions: dict[str, float]
    joint_velocities: dict[str, float] | None
    joint_efforts: dict[str, float] | None
    gripper_widths: dict[str, float]
    health: DriverHealth


@dataclass(frozen=True)
class RobotCommand:
    timestamp_s: float
    joint_positions: dict[str, float]
    gripper_widths: dict[str, float]
```

Named values are used at API boundaries. Backends may convert them to ordered
arrays only after validating against the model's declared canonical order.

### Robot driver

```python
class RobotDriver(Protocol):
    @property
    def capabilities(self) -> DriverCapabilities: ...
    def connect(self) -> None: ...
    def read_state(self) -> RobotState: ...
    def command(self, command: RobotCommand) -> None: ...
    def hold(self) -> None: ...
    def move_home(self) -> None: ...
    def close(self) -> None: ...
```

Vendor units, SDK objects, transport threads, watchdogs, and error codes stay
inside the driver implementation. A software `hold()` is not a replacement for
the robot's physical emergency stop.

### Kinematics backend

```python
class KinematicsBackend(Protocol):
    @property
    def joint_names(self) -> tuple[str, ...]: ...
    def forward(self, state: JointState) -> dict[str, Pose]: ...
    def solve(
        self,
        seed: JointState,
        targets: dict[str, Pose],
        dt_s: float,
    ) -> IKResult: ...
```

`IKResult` must include the solution, convergence state, per-role pose error,
limit activity, and diagnostics. The backend consumes explicit groups and
roles; it never searches for name prefixes.

### Teleoperation controller

The shared controller owns:

- initial robot/controller anchors;
- local-relative and absolute-table retargeting;
- clap activation and pause state;
- tracking-loss recovery;
- inactive-side behavior;
- homing transitions;
- stale-state rejection;
- production of task-role targets.

It is a deterministic state machine with no CAN, SDK, MuJoCo, or Viser imports.
Real and simulated runners only provide observations, time, and backend I/O.

### Safety filter

The filter runs after IK and before every backend command. It owns:

- position-limit enforcement;
- velocity and acceleration limits using measured `dt_s`;
- stale-command and stale-state rejection;
- non-finite value rejection;
- optional workspace and self-collision checks;
- transition-to-hold policy on failure.

Robot-specific limits come from the model/deployment profile. Transport-level
watchdogs remain a second protection layer inside each driver.

## 5. Versioned Robot Profile

The profile schema must be validated before loading model assets. A possible
version 1 shape is:

```yaml
schema_version: 1
name: example_bimanual

model:
  urdf: package://example_robot/robot.urdf
  package_roots: []
  joint_order: [a1, a2, a3, b1, b2, b3]

joint_groups:
  arm_a: [a1, a2, a3]
  arm_b: [b1, b2, b3]

task_roles:
  left:
    joint_group: arm_a
    base_link: base_a
    tcp_link: tool_a
  right:
    joint_group: arm_b
    base_link: base_b
    tcp_link: tool_b

grippers:
  left:
    type: parallel
    joints: [finger_a]
    width_range_m: [0.0, 0.07]
  right:
    type: parallel
    joints: [finger_b]
    width_range_m: [0.0, 0.07]

home:
  joint_positions:
    a1: 0.0
    a2: 0.2
    a3: -0.2
    b1: 0.0
    b2: 0.2
    b3: -0.2

kinematics:
  backend: jaxls
  position_weight: 20.0
  orientation_weight: 1.0
  posture_weight: 0.05

robot_tool_calibrations:
  meta_piper_gripper_v1:
    gripper: piper_parallel_v1
    controller: meta
    calibration: package://example_robot/calibration/meta_tcp.yaml

simulation:
  backend: mujoco
  model: package://example_robot/robot.xml
  actuator_map:
    a1: actuator_1
    a2: actuator_2
  collision_policy: package://example_robot/collision.yaml
```

Separate single-arm model instances must be composable into the two task roles.
A combined bimanual URDF remains supported, but is no longer mandatory.

### Discovery

- Built-in profiles are loaded with `importlib.resources`.
- User profiles may be supplied through `--robot-profile` and a documented
  user configuration directory.
- Driver factories are discovered through Python entry points, for example
  `handumi.robot_drivers`.
- CLI validation happens after discovery; parsers do not use a static tuple of
  robot names.
- Error messages list discovered profiles/plugins and their validation errors.
- The installed wheel is tested, not only the source checkout.

Adding a configuration-only model must not require a Python edit. Adding new
hardware requires an independently packaged driver plugin, not a vendor branch
inside `teleop_real.py`.

## 6. Dataset and Provenance Rules

### Raw capture metadata

Raw datasets store facts about capture:

- source device and schema version;
- controller poses and conventions;
- capture profile and robot-tool calibration snapshots/hashes;
- robot, gripper/tool, controller-mount, and side identities associated with the
  selected Controller-to-TCP calibration;
- cameras, gripper inputs, tracking health, and timestamps;
- table/world calibration available during recording.

The target embodiment is optional intent, not a required property of the raw
trajectory. `handumi-record` must work without loading robot assets.

### Derived replay/conversion metadata

Derived artifacts additionally store:

- selected model and deployment profile hashes;
- retargeting mode and spatial transforms;
- IK and safety configuration;
- joint/gripper mapping;
- solver diagnostics and fidelity metrics.

Conversion never mutates the source dataset. The same raw episode must be
retargetable to multiple robot embodiments.

## 7. Migration Plan

Each phase must be independently releasable and keep the Piper path working.

### Phase 0 — Characterize Existing Behavior

- Add golden metadata tests for the current raw Meta capture.
- Record Piper replay baselines: target TCPs, joint outputs, IK errors, and
  gripper commands for representative episodes.
- Add fake tracking and fake Piper transport fixtures.
- Document current public CLI flags and accepted profile fields.

Exit criteria:

- Baselines fail on unintended behavior changes.
- Tests require no robot hardware.

### Phase 1 — Make Raw Capture Robot-Independent

- Replace the semantic meaning of `--robot` in recording with optional
  `--intended-embodiment`.
- Keep `--robot` as a deprecated compatibility alias for one migration window.
- Do not load or snapshot a robot profile unless intent is explicitly supplied.
- When an intended embodiment is supplied, resolve its robot-tool calibration
  and snapshot the exact Controller-to-TCP transforms and provenance.
- Permit uncalibrated raw controller-pose recording without a selected robot,
  while marking calibrated TCP output unavailable.
- Move target-model snapshots to derived replay/conversion artifacts.

Exit criteria:

- `handumi-record` runs with no robot selected or installed.
- Existing recording commands still work with a deprecation warning.
- The raw observation/action schema is unchanged.

### Phase 2 — Version Profiles and Remove Naming Assumptions

- Introduce a validated `schema_version: 1` model.
- Declare canonical order, joint groups, task roles, base links, TCP links, and
  gripper mappings explicitly.
- Add versioned robot-tool calibration references keyed by robot, gripper/tool,
  controller mount, and side.
- Add a version-0 adapter for current Piper and Axol YAML files.
- Replace `EMBODIMENT_NAMES` with dynamic discovery in record, replay,
  conversion, and teleoperation CLIs.
- Load all shipped assets through package resources.
- Add a test fixture whose joint and link names contain no left/right prefixes.
- Add a composed two-single-arm fixture.

Exit criteria:

- Piper and Axol load through the versioned schema.
- Core kinematics obtains no semantics from string prefixes.
- A newly installed model profile is visible without editing HandUMI source.
- All profiles and referenced assets load from an installed wheel.

### Phase 3 — Introduce the Driver Contract and Adapt Piper

- Add generic state, command, health, and capability types.
- Wrap `PiperCanEnvironment` behind `RobotDriver` without rewriting the tested
  transport implementation.
- Move Piper unit conversion and error mapping fully behind the adapter.
- Move Piper connection defaults to a driver/deployment profile and local rig
  configuration.
- Give driver plugins an optional setup/diagnostic entry point.
- Keep vendor dependencies optional.

Exit criteria:

- `teleop_real.py` has no Piper imports, unit conversions, or robot-name branch.
- A fake driver can run the real teleoperation runner.
- Piper produces equivalent commands to the Phase 0 baseline.

### Phase 4 — Share the Teleoperation State Machine

- Extract activation, anchoring, retargeting, tracking recovery, inactive-side,
  pause, hold, and homing logic into a pure controller.
- Drive that controller with both the real and simulation runners.
- Use an injected monotonic clock for deterministic tests.
- Make operator events and backend failures explicit inputs.

Exit criteria:

- The same transition tests run against fake real and simulated backends.
- Real/sim differences are limited to I/O, visualization, and physics.
- Tracking loss always leads to the tested hold/recovery behavior.

### Phase 5 — Generalize IK and Safety

- Build IK variables and costs from declared groups and roles.
- Support combined and composed robot models.
- Replace per-frame joint deltas with velocity/acceleration limits using `dt_s`.
- Either implement declared posture/manipulability weights or remove them from
  public configuration until supported.
- Return structured per-role IK diagnostics.
- Apply the safety filter before both simulated and real commands.

Exit criteria:

- No `_side_indices` or prefix-derived joint grouping remains.
- Different control rates produce equivalent physical motion limits.
- Invalid, stale, or unsafe solutions never reach a backend.

### Phase 6 — Make Simulation Profile-Driven

- Introduce a simulation backend boundary using the same state/command types.
- Move actuator names, gripper coupling, base placement, camera defaults, and
  collision masks into profiles or robot-owned assets.
- Remove Piper-specific collision mutation from generic MuJoCo code.
- Keep Viser as a renderer that consumes the declared model and state.
- Validate actuator and joint mappings at startup.

Exit criteria:

- Piper, Axol, and the prefix-free fixture run without simulator name branches.
- A profile cannot silently command the wrong actuator.
- Collision behavior is explicit and testable per model.

### Phase 7 — Prove the Boundary with a Second Hardware Family

Do not stabilize the driver API using Piper alone. Select one materially
different integration as the proof backend:

1. **I2RT YAM direct driver**: useful for exercising a different CAN SDK while
   retaining joint-position control and published URDF/MJCF assets.
2. **TRLC DK1 adapter**: useful for exercising interoperability with a LeRobot
   robot plugin instead of embedding another serial protocol in HandUMI.

Axol remains a valuable second model/simulation test, but it does not by itself
prove the real-driver boundary.

- Implement the selected adapter outside core orchestration.
- Map its state and commands to canonical SI units.
- Declare unsupported capabilities instead of emulating them silently.
- Run the same driver contract and teleoperation state-machine tests as Piper.
- Feed any required API changes back into the contract before declaring it
  stable.

Exit criteria:

- Piper and the second hardware family use the same core runner unchanged.
- Adding the second driver required no robot-name branch in core code.
- Disconnect, timeout, hold, home, and recovery behavior pass contract tests.

### Phase 8 — Deprecate Compatibility Paths

- Replace legacy Controller-to-TCP keys with the versioned robot-tool
  calibration schema while retaining the association with each robot profile.
- Remove the version-0 profile adapter after the documented migration window.
- Remove the recording `--robot` alias after users migrate.
- Delete `EMBODIMENT_NAMES` and obsolete Piper-specific generic helpers.
- Rewrite `docs/add_new_embodiment.md` around models, drivers, and deployments.
- Add migration examples for Piper, Axol, and the proof backend.

Exit criteria:

- No generic module imports an optional vendor dependency.
- No current documentation instructs users to edit a central robot-name list.
- All deprecations have actionable replacement messages.

## 8. Recommended Pull Request Slices

Keep reviews narrow and preserve behavior between slices:

1. Characterization tests and raw-capture metadata contract.
2. Optional intended embodiment in recording.
3. Versioned model schema, validator, and version-0 adapter.
4. Dynamic discovery and installed-wheel packaging test.
5. Generic driver types and fake driver.
6. Piper driver adapter and generic real runner.
7. Shared teleoperation controller.
8. Time-aware safety and explicit-group IK.
9. Profile-driven simulation.
10. Second real-driver proof and compatibility cleanup.

Do not combine the model-schema migration, Piper driver migration, and shared
controller extraction in one pull request. Their baselines must isolate model,
transport, and behavior regressions.

## 9. Compatibility Policy

- Existing raw datasets remain readable and are never rewritten in place.
- Existing Piper commands remain functional during the migration window.
- Version-0 profiles are translated with explicit warnings, never guessed
  silently.
- Calibration files and hashes used for an output remain in its provenance.
- Missing optional vendor packages fail only when that driver is selected.
- A deployment must fail closed when required transforms or joint mappings are
  absent or ambiguous.
- New limits may be stricter than old limits; any intentional motion change must
  be called out and validated on simulation before hardware.

## 10. Acceptance Criteria

The project is robot-agnostic for the initial scope when all are true:

- Raw recording works without selecting or installing a robot model.
- The same raw episode can be converted to Piper and another model without
  modifying the source dataset.
- Every calibrated episode identifies and snapshots the robot, gripper/tool,
  controller mount, side, and Controller-to-TCP calibration used at capture.
- Replay uses the dataset's source robot-tool calibration; selecting a new
  target embodiment does not overwrite it.
- Model profiles declare all groups and frames; no prefix convention is needed.
- A new configuration-only model needs no core Python edit.
- A new real robot is added through a driver plugin and deployment, with no core
  robot-name branch.
- Real and simulated teleoperation use the same state machine and safety filter.
- Piper and one materially different real-driver family pass the same contract
  tests.
- Driver boundaries use SI units and timestamped named state/command values.
- MuJoCo behavior is controlled by explicit model mappings and collision policy.
- All built-in profiles and assets load from the installed wheel.
- Existing Piper replay stays within the agreed Phase 0 joint/TCP tolerances.

## 11. External Integration References

- I2RT YAM documentation: <https://doc.i2rt.com/products/yam>
- TRLC DK1 LeRobot-compatible implementation:
  <https://github.com/robot-learning-co/trlc-dk1>
- Hugging Face LeRobot robot interface and plugin ecosystem:
  <https://github.com/huggingface/lerobot>

These are candidate integration boundaries, not dependencies of the core
architecture.
