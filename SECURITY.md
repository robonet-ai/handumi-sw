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

## Network boundary

Quest pose TCP on port 65432 and clock-sync UDP on port 42000 are
unauthenticated plaintext protocols. Use them only on a trusted, isolated local
network. Do not forward either port to the public internet. Apply host/network
firewall rules that restrict traffic to the expected workstation and headset.

Rerun and Viser should bind to loopback by default. A LAN-visible viewer bind
is an explicit operator decision and can expose camera frames, body motion,
robot state, filenames, and calibration context to other network users. Use an
authenticated tunnel or equivalently controlled network when remote viewing is
required. HandUMI does not add transport encryption or authentication.

## Privacy and retention

Obtain informed consent before recording a person or an environment. Treat raw
and derived controller, HMD, body, camera, calibration, and timing data as
sensitive. Before capture, define who can access it, the retention period, and
how deletion requests are handled. Store datasets on access-controlled media,
remove rejected/interrupted episodes according to the study policy, and review
every dataset and release artifact before publication. Do not assume that
removing names anonymizes motion, imagery, device identifiers, or room details.

## Physical robots

Real-robot operation requires an independent emergency stop, hardware and
software watchdogs, conservative workspace/joint/velocity/acceleration limits,
collision controls, and a preview or dry-run process appropriate to the robot.
Keep people outside the active workspace unless an approved laboratory safety
procedure says otherwise. Estimated body, CoM, contact, and support state must
never enable, disable, or substitute for a safety interlock.
