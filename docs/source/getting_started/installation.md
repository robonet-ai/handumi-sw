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
handumi record --help
```

`install.sh` creates the virtual environment, runs `uv sync`, and builds the
XRoboToolkit SDK needed for PICO. Use `--skip-xrt` when the setup only uses
Meta Quest. It also creates the ignored machine-local `configs/rig.yaml` from
`configs/rig.example.yaml` without overwriting an existing rig configuration.
Activating the environment loads command and option completion for Bash, Zsh,
or Fish; for example, `handumi re<Tab>` offers `record` and `replay`.
`hu` is an equivalent short alias for the complete CLI, including help and
completion, so `hu record` and `handumi record` behave identically.

For installations that do not use `install.sh`, enable completion in the
current shell with one of:

```bash
# Bash
eval "$(handumi completion bash)"

# Zsh
eval "$(handumi completion zsh)"

# Fish
handumi completion fish | source
```

## Optional robot and simulation profiles

The base environment does not install manufacturer SDKs. Select only the
profiles needed on the workstation:

```bash
bash install.sh --skip-xrt --sim --robot openarmv1
# Or manage profiles directly after installing system prerequisites:
uv sync --extra sim
uv sync --extra piper
uv sync --extra openarm
uv sync --extra cuda --extra sim
```

`install.sh --robot openarmv1` installs the official Ubuntu system packages
before building the pinned Python binding. The equivalent manual sequence is:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:openarm/main
sudo apt update
sudo apt install -y libopenarm-can-dev openarm-can-utils
uv sync --extra openarm
```

Simulation does not require `piper_sdk` or `openarm_can`.
