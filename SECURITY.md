# Security Policy

## Supported releases

HandUMI is currently a research preview. Security fixes are applied to the
latest tagged release and the default branch; older snapshots may not receive
backports.

## Reporting a vulnerability

Do not open a public issue for a vulnerability, credential exposure, unsafe
robot-command path, or privacy leak. Use GitHub's **Report a vulnerability**
feature in the repository Security tab. Include the affected commit/version,
reproduction conditions, impact, and any suggested mitigation. Maintainers
will acknowledge a complete report within seven days and coordinate disclosure
after a fix or documented mitigation is available.

Do not include real participant motion/video, secrets, device identifiers,
private network addresses, or signing material in a report. Use synthetic or
redacted evidence.

## Operational boundary

Body, CoM, contact, and support outputs are not safety-rated. Keep robot safety
independent of headset/body estimates, validate all trajectories, enforce
hardware limits, and retain a physical emergency stop. Treat headset camera,
motion, calibration, and environment captures as sensitive human-subject data.
