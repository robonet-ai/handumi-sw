# Add a New Robot Embodiment

Ultima modificacion: 2026-07-11 19:51:01 -05 -0500

Robots are config-driven: one YAML under `configs/robots/` describes the
embodiment, and the registry (`src/handumi/robots/registry.py`) builds the
URDF model, bimanual IK solver, and Viser sim from it. No per-robot Python
code is needed.

## 1. Drop the assets

Put the URDF and its meshes under `assets/<robot>/`. Requirements:

- **Bimanual, single file**: both arms in one URDF, with actuated joint
  names prefixed `left_` / `right_` (the registry derives per-arm joint
  lists and the command size from those prefixes).
- **An EE link per side**: the link the IK targets. Prefer an explicit TCP
  link at the gripper tip (see `left_tcp`/`right_tcp` in
  `assets/piper/piper.urdf`) over the wrist flange, so replayed TCP
  trajectories land on the tip.
- `package://PKG/...` mesh paths are fine — `pkg_root` below resolves them.

## 2. Write the config

Create `configs/robots/<robot>.yaml`:

```yaml
kind: myrobot
urdf: assets/myrobot/myrobot.urdf
pkg_root: assets/myrobot          # resolves package:// mesh references
ee_links:
  left: left_tcp                  # IK target link per side
  right: right_tcp
home_q: [0.0, 0.0, ...]           # one value per actuated joint (or [] for zeros)
ik_weights:                       # optional; defaults shown in registry.py
  pos: 100.0
  ori: 15.0
  rest: 2.0
  max_reach: 0.45                 # optional workspace clamp (meters)
```

Existing examples: [configs/robots/piper.yaml](../configs/robots/piper.yaml),
[configs/robots/axol.yaml](../configs/robots/axol.yaml).

## 3. Register the name

Add the name to `EMBODIMENT_NAMES` in `src/handumi/robots/registry.py`
(and pick its default Viser port there if it needs one).

## 4. Verify

```bash
# solve + view a recorded episode on the new robot
handumi-replay-in-sim --repo-id <dataset> --robot myrobot

# or follow live tracking with it
handumi-teleop-sim --device meta --robot myrobot
```

The replay reports per-frame EE pose errors — large errors usually mean
wrong `ee_links`, a bad `home_q`, or IK weights that need tuning.
