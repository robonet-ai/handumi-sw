# YAM bimanual assets

This directory contains a simulation/replay model for two standard I2RT YAM
follower arms equipped with `linear_4310` parallel grippers.

Source: `i2rt-robotics/i2rt` commit
`ac096928d6899ddf852a71c5e8fbaa6055cd9745`, licensed under MIT. The arm
kinematics come from `i2rt/robot_models/arm/yam/yam.urdf`; the visual meshes
come from the current YAM and `linear_4310` asset directories in that commit.

`yam_bimanual.urdf` makes only the adaptations required by HandUMI:

- two copies are namespaced as `left_` and `right_`;
- the upstream zero-range root joint is represented as a fixed joint;
- both bases are placed 0.60 m apart in a +X-right, +Y-forward, +Z-up world;
- the stale upstream mesh paths are replaced with the current I2RT assets;
- `link2` and `link3` are split into arm/casing visual meshes without changing
  their triangles, so YAM's white links and black motor housings can use
  separate URDF materials;
- fixed TCP links use I2RT's `grasp_site` offset of 0.1347 m;
- the two finger slides per arm expose the full 0.096 m nominal stroke.

This is currently a kinematic model. It does not register an I2RT hardware
backend or a bimanual MuJoCo contact model.
