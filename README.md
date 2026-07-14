# HandUMI

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/BrikHMP18/HandUMI"><img src="https://img.shields.io/badge/Hardware-HandUMI-4c8bf5.svg" alt="HandUMI hardware"></a>
  <a href="https://robonet-ai.github.io/handumi-sw/"><img src="https://img.shields.io/badge/Docs-HandUMI-AF0000.svg" alt="HandUMI documentation"></a>
</p>

[HandUMI](https://github.com/BrikHMP18/HandUMI) is a hand-worn interface for collecting robot-free bimanual demonstrations. This repository contains its synchronized data collection, calibration, validation, replay, teleoperation, and robot-retargeting software.

## Documentation

**[Read the HandUMI documentation](https://robonet-ai.github.io/handumi-sw/)**

- [Installation](docs/source/getting_started/installation.md)
- [Setup and calibration](docs/source/setup.md)
- [Record demonstrations](docs/source/record.md)
- [Teleoperation](docs/source/teleoperation.md)
- [Dataset pipeline](docs/source/workflows/datasets.md)
- [Troubleshooting](docs/source/troubleshooting.md)
- [Add a new robot embodiment](docs/source/development/new_embodiment.md)

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
git clone https://github.com/robonet-ai/handumi-sw.git
cd handumi-sw
bash install.sh
source .venv/bin/activate
handumi-record --help
```

PICO support is installed by default. Use `bash install.sh --skip-xrt` for a Meta Quest-only workstation.

## Core Workflow

```text
tracking + cameras + gripper widths
                ↓
     synchronized raw dataset
                ↓
       validate → convert/replay
```

Raw captures remain robot-agnostic. Robot configuration and physical controller-to-TCP calibration are fingerprinted in dataset metadata so later conversion remains reproducible.

## Supported Scope

- Tracking: PICO through XRoboToolkit and Meta Quest through
  [HandUMI Quest App](https://github.com/robonet-ai/handumi-quest-app).
- Robot models and simulation: Piper and Axol.
- Real-robot teleoperation: AgileX Piper.
- Dataset format: LeRobot-compatible synchronized captures.

## Safety

This is research software. Preview and validate trajectories before commanding physical robots, keep an emergency stop accessible, and enforce the robot's joint, velocity, acceleration, workspace, and collision limits.

## Credits

HandUMI builds on UMI, HandUMI Quest App, XRoboToolkit, LeRobot, PyRoki,
Viser, Rerun, and MuJoCo. See the [documentation](https://robonet-ai.github.io/handumi-sw/)
and [LICENSE](LICENSE) for attribution and third-party licensing details.

Project lead and original hardware design: [BrikHMP18](https://github.com/BrikHMP18). Core software contributors include [Leonardo Pérez](https://github.com/leoperezz), [Raul Bastidas](https://github.com/RAUL-BASTIDAS), [Mitshell Ramos](https://github.com/mbrq13), and [Alvaro Mendoza-Li](https://github.com/alvax64).

## License

Original HandUMI software and documentation are licensed under the [Apache License 2.0](LICENSE). Dataset, hardware, headset application, robot firmware, and trademark licenses remain separate.
