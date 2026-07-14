# Meta Quest Tracking Setup

Streams controller/HMD poses over TCP/JSON + UDP time-sync. The legacy
**YubiQuestApp** remains supported without payload changes. The separate
**HandUMI Body Probe** app adds an additive `tracking_packet_v2` envelope and
compact 70/84-joint body channel. For tabletop
capture, rigidly mount the Quest with both controllers inside its camera
coverage. For portable legacy capture, secure it rigidly to the chest; do not
leave it free to swing from the neck. One controller is mounted on each gripper.
No XRoboToolkit needed; install the repo with
`bash install.sh --skip-xrt`.

Full-body qualification is a separate head-worn mode. Meta body tracking must
be tested with the headset worn normally and a floor-level acquisition space;
chest/neck-mounted data is not evidence that the platform body model works.
Quest HMD, controller, hand, and body poses are platform-provided estimates,
not direct physical measurements. Anatomical center of mass is not measured by
the Quest runtime.

## Install (one-time)

1. Enable Developer Mode: free org at <https://developers.meta.com/>, then
   Meta Horizon app → *Devices → headset → Developer Mode → on*.
2. Connect USB-C, authorize: `adb devices` → accept the prompt in the
   headset until it lists `device`.
3. Install the app (into gitignored `external_dependencies/`):

   ```bash
   mkdir -p external_dependencies/quest-app && cd external_dependencies/quest-app
   wget https://releases.dev.airoa.io/yubi/quest-app/yubi-quest-app-v0.1.0.apk
   adb install -r yubi-quest-app-v0.1.0.apk && cd -
   ```

## Connect (per network change)

Streaming is over Wi-Fi (USB only installs; unplug after). Both on the same
network, then pin the Quest IP in `configs/tracking_meta_quest.yaml`:

```bash
adb shell ip route     # prints ... src <quest-ip>
```

## Smoke-test

Launch YubiQuestApp (Library → Unknown Sources). It must stay foreground and
the proximity sensor must remain active while the Quest is mounted. Wake both
controllers and keep them in view of the headset cameras.

```bash
python -m handumi.tracking.meta_quest --config configs/tracking_meta_quest.yaml
```

Good = steady `fps` (~120) and both `trk=1` with positions that move.
No Quest at hand? Fake one: `python -m handumi.tracking.mock_quest_sender`
(+ receiver with `--quest-ip 127.0.0.1`).

## Full-body platform probe

The workstation probe records each sender packet unchanged and adds PC receive
time plus the current UDP clock offset and RTT. It is intended for qualifying a
diagnostic Quest build that reports OpenXR body extension, calibration,
fidelity, joint flag, and source-time evidence. It does not add body tracking to
the legacy APK.

```bash
handumi-quest-probe capture \
  --config configs/tracking_meta_quest.yaml \
  --duration-s 300 \
  --adb-health \
  --output artifacts/quest-probe/neutral-head-worn

handumi-quest-probe analyze \
  artifacts/quest-probe/neutral-head-worn/quest_packets.jsonl
```

The output directory contains the append-only `quest_packets.jsonl`, a separate
`session_manifests.jsonl`, `capture_context.json`, and `summary.json`. With
`--adb-health`, it also contains timestamped `quest_health.jsonl` battery,
thermal, CPU, memory, process, and lifecycle snapshots plus
`quest_logcat.txt`. The analyzer excludes manifests from pose-rate and loss
calculations; distinguishes receive/render rate from distinct body source-time
rate; reports gaps, duplicates, resets, and reordering; and calculates raw
location-flag valid/tracked percentages for every joint. Device-clock and body
sample-age statistics remain unavailable until UDP synchronization succeeds.

A mock-sender run validates the workstation transport and analysis only. It
cannot establish runtime extension support, body accuracy, or a zero-loss Quest
session unless the sender includes a real monotonic `seq`.

## Production tracking packet API

`MetaQuestTrackingProvider` and `PicoTrackingProvider` retain their legacy
`latest()` controller interface. New acquisition code can instead use
`latest_packet()` for UI/control snapshots and `drain_packets()` for every
queued sample. The queue is bounded by `streams.packet_queue_size`; overflow,
malformed frames, unsupported versions, source gaps, duplicates, and reordering
are exposed through packet-stream diagnostics. Raw writers should drain the
FIFO with `drain_tracking_packets_jsonl()` so source fields unknown to this
software are preserved.

The common packet has optional HMD, controller, body, hand, and external
tracker channels. Joint location flags remain exact and invalid joints are not
filled from prior poses. Source time, mapped PC time, receive time, clock
offset, RTT, uncertainty, provenance, and timestamp quality are separate
fields. Meta body timestamps currently carry `DIAGNOSTIC_ONLY`: do not treat
receive time as body sample time or use this body channel for precision sensor
fusion until the source-time provenance is qualified.

## Troubleshooting

- **Port refused / no frames** — app not foreground or headset not "worn";
  `quest_ip` stale (DHCP); different networks.
- **`trk=0` / frozen poses** — controllers asleep or out of camera view.
- **`INSTALL_FAILED_UPDATE_INCOMPATIBLE`** —
  `adb uninstall com.UnityTechnologies.com.unity.template.urpblank`, reinstall.

---

Next: hardware + width calibration ([README_gripper_width.md](README_gripper_width.md)),
mount calibration ([README_tcp_offset.md](README_tcp_offset.md)), then
record per [README.md](../README.md).
