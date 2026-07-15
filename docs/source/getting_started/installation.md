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
