# Galaxea R1 Lite assets

Simulation/replay model for the Galaxea R1 Lite mobile bimanual manipulator.

Source: [Galaxea ATC V2.3.0 SDK](https://docs.galaxea-dynamics.com/R1Pro/en/docs/2026/development/r1pro_get_sdk/)
(`atc_system-V2.3.0-20260403_12_02_32_aarch64/install/r1lite_urdf/share/r1lite_urdf/`).
The kinematics, visual meshes, and simplified collision meshes come from the
vendor `r1lite_2026.urdf` package layout.

`r1lite.urdf` makes only the adaptations required by HandUMI:

- `package://r1lite_urdf/meshes/`;
- `package://r1lite_urdf/collision_meshes/` on the seven
  vendor collision links;

This is currently a kinematic model for simulation and replay. It does not
register a Galaxea hardware backend.
