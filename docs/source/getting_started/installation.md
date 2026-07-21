# Installation

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```bash
git clone https://github.com/robonet-ai/handumi-sw.git
cd handumi-sw
bash install.sh              # PICO support included
# bash install.sh --skip-xrt # Meta Quest only
source .venv/bin/activate
```

Check:

```bash
python --version
handumi-record --help
```

`install.sh` creates the virtual environment, runs `uv sync`, and builds the
XRoboToolkit SDK needed for PICO. Use `--skip-xrt` when the setup only uses
Meta Quest. It also creates the ignored machine-local `configs/rig.yaml` from
`configs/rig.example.yaml` without overwriting an existing rig configuration.

## Optional robot and simulation profiles

The base environment does not install manufacturer SDKs. Select only the
profiles needed on the workstation:

```bash
bash install.sh --skip-xrt --sim --robot openarmv1
# Or manage profiles directly after installing system prerequisites:
uv sync --extra sim
uv sync --group piper-source
uv sync --group openarm-source
uv sync --extra cuda --extra sim
```

`install.sh --robot openarmv1` installs the official Ubuntu system packages
before building the pinned Python binding. The equivalent manual sequence is:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:openarm/main
sudo apt update
sudo apt install -y libopenarm-can-dev openarm-can-utils
uv sync --group openarm-source
```

Simulation does not require `piper_sdk` or `openarm_can`.

The base wheel deliberately contains no unpublished Git dependencies and does
not pull PyTorch/CUDA through LeRobot. Full recording and IK are source-release
integrations: the default source `uv sync` selects the pinned `recording-source`
and `ik-source` groups, with PyTorch resolved from the configured CPU-only
index. `piper-source`, `openarm-source`, and `pico-source` are explicit groups.
This separation keeps a clean wheel install bounded while the unpublished
PyRoki/JAXLS and manufacturer SDK dependencies remain ineligible for PyPI.
