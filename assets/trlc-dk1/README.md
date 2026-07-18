# TRLC-DK1 robot assets

The single-arm follower URDF and meshes originate from the Apache-2.0
[`robot-learning-co/trlc-dk1`](https://github.com/robot-learning-co/trlc-dk1)
project.

HandUMI uses `TRLC-DK1-Bimanual.urdf`, which contains two namespaced copies
of the follower under a shared world link.

The current layout places the arm bases 0.60 m apart laterally.
This is a provisional simulation layout, not a measurement of a physical
deployment. Update the generator's `--base-separation-m` value and
`configs/calibration/trlc_dk1_table.yaml` from physical measurements before
using absolute placement on hardware.
