# Piper Hardware Setup

This procedure prepares two physical AgileX Piper arms for HandUMI real
teleoperation. Complete the robot-independent
[HandUMI Setup and Calibration](../setup.md) first.

## Safety and prerequisites

Before connecting or commanding the arms:

- Clear the complete arm workspace and keep the emergency stop reachable.
- Power both arms, but stop every other process that may use their CAN buses.
- Install the Piper backend with `uv sync --extra piper`.
- Connect one USB-to-CAN adapter per arm.
- Verify tracking and motion mapping in simulation before enabling hardware.

## Install and map the CAN adapters

Run the guided hardware setup:

```bash
uv sync --extra piper
handumi-setup-hardware --robot piper --device meta \
  --skip-feetech-map --skip-feetech-calibration
```

The wizard maps the right Piper adapter first and the left adapter second. It
stores the machine-local result under `robots.piper.can` in
`configs/rig.yaml`. Follow the prompts to disconnect and reconnect adapters so
that each physical side is identified correctly.

Use `--skip-can-map` only after verifying an existing mapping. Rerun the wizard
whenever adapters, USB ports, or arm assignments change.

## Verify CAN and troubleshoot the mapping

Check that both arms are powered, both adapters are present, and no other
process owns the CAN interfaces. If an interface is down or bus-off, stop
teleoperation, inspect power and wiring, and rerun the guided setup from the
previous section.

Do not continue to real motion until the wizard identifies both physical sides
and communication is stable.

## First real teleoperation

Start with simulation and the same robot profile:

```bash
handumi-teleop-sim --device meta --robot piper --space-start
```

After tracking, calibration, and simulated motion behave correctly, validate
one physical arm first:

```bash
handumi-teleop-real --device meta --robot piper --side right
```

Keep the emergency stop reachable and confirm that the right controller moves
only the right arm. Stop and correct the CAN mapping if the wrong side moves.
Validate the left side separately before enabling both arms:

```bash
handumi-teleop-real --device meta --robot piper --side left
handumi-teleop-real --device meta --robot piper --side both
```

For shared controls, safety behavior, and tracking semantics, continue with
[Teleoperation](../teleoperation.md). For common failures, see
[Troubleshooting](../troubleshooting.md).
