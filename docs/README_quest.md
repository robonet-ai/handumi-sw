# Meta Quest Tracking Setup

Streams controller/HMD poses over TCP/JSON + UDP time-sync using the
prebuilt **YubiQuestApp** from [yubi-sw](https://github.com/airoa-org/yubi-sw).
For tabletop capture, rigidly mount the Quest with both controllers inside its
camera coverage. For portable capture, secure it rigidly to the chest; do not
leave it free to swing from the neck. One controller is mounted on each
gripper. No XRoboToolkit needed; install the repo with
`bash install.sh --skip-xrt`.

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
