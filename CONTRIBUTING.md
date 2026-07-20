# Contributing to HandUMI

HandUMI welcomes focused issues and pull requests. By contributing, you agree
that your contribution is licensed under Apache-2.0.

## Development

1. Install with `bash install.sh` (or `bash install.sh --skip-xrt` for a Meta-only workstation).
2. Run `uv lock --check` and `uv run pytest -q` before opening a pull request.
3. Build docs with `uv run --with-requirements docs/requirements.txt make -C docs html`.
4. Keep hardware tests explicit; never make CI depend on an attached headset,
   camera, CAN bus, servo, or robot.

Preserve raw observations, masks, timing, calibration, and provenance. New
estimated signals must fail closed when required calibration is absent and
must not be described as measured. Do not submit participant recordings,
credentials, signing keys, machine-local `configs/rig.yaml`, calibration
caches, or captured `outputs/`/`artifacts/`.

## Scientific claims

This release is a research preview. Changes may improve visualization without
proving anatomical accuracy or timing. Full-body pose, CoM, contact, and
support claims require the repository's participant-level ground-truth
validation protocol and documented evidence.
