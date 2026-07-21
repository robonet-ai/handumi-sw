# HandUMI

```{warning}
HandUMI is a research preview. Full-body pose, center of mass, contact,
support, and profile-constrained skeleton outputs are experimental estimates
until the documented ground-truth validation gates pass. They are not
anatomical, synchronization-grade, medical, ergonomic, or production-safety
measurements.
```

Collect robot-free bimanual demonstrations once with HandUMI, then validate,
retarget, and reuse them across different bimanual arms with parallel grippers.

```{image} _static/HandUMI.png
:alt: HandUMI hardware
:class: handumi-cover
:width: 100%
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting_started/installation
setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Core Workflows

teleoperation
record
workflows/replay_in_sim
   workflows/datasets
   workflows/dataset_compatibility
   workflows/capture_reliability
workflows/body_tracking
workflows/visualization
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Physical Robots

physical_robots/piper_setup
physical_robots/openarm_v1_setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Help

troubleshooting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

   development/new_embodiment
   development/test001_protocol_template
development/release_checklist
```
