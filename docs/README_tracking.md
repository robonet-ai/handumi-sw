# Tracking Setup (Meta Quest / PICO)

One-time setup per tracking device. Quest streams controller/HMD poses over
TCP/JSON + UDP time-sync; PICO streams via XRoboToolkit. Either way the
device is worn on the neck as a tracking base, one controller mounted on
each gripper.

---

## Meta Quest

Uses the prebuilt **YubiQuestApp** from
[yubi-sw](https://github.com/airoa-org/yubi-sw). No XRoboToolkit needed —
install the repo with `bash install.sh --skip-xrt`.

### Install (one-time)

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

### Connect (per network change)

Streaming is over Wi-Fi (USB only installs; unplug after). Both on the same
network, then pin the Quest IP in `configs/tracking_meta_quest.yaml`:

```bash
adb shell ip route     # prints ... src <quest-ip>
```

### Smoke-test

Launch YubiQuestApp (Library → Unknown Sources) — it must stay foreground
with the headset "worn" (cover the proximity sensor if it hangs on the
neck). Wake both controllers, keep them in view of the headset cameras.

```bash
python -m handumi.tracking.meta_quest --config configs/tracking_meta_quest.yaml
```

Good = steady `fps` (~120) and both `trk=1` with positions that move.
No Quest at hand? Fake one: `python -m handumi.tracking.mock_quest_sender`
(+ receiver with `--quest-ip 127.0.0.1`).

### Troubleshooting

- **Port refused / no frames** — app not foreground or headset not "worn";
  `quest_ip` stale (DHCP); different networks.
- **`trk=0` / frozen poses** — controllers asleep or out of camera view.
- **`INSTALL_FAILED_UPDATE_INCOMPATIBLE`** —
  `adb uninstall com.UnityTechnologies.com.unity.template.urpblank`, reinstall.

---

## PICO

Uses **XRoboToolkit**: a PC service (workstation) + headset app + Python
SDK. `bash install.sh` (without `--skip-xrt`) builds the SDK.

### Install (one-time)

1. PC service (Ubuntu x86_64; other platforms in the
   [releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)):

   ```bash
   wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
   sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
   ```

2. Headset app: in the PICO browser, download and install
   [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-PICO/releases)
   (needs Developer Mode on; app lands in Library → Unknown).
3. USB mode (default transport): enable USB debugging
   (*Settings → Developer*), connect USB-C, authorize via `adb devices`.

### Connect (per session)

1. Start the PC service: `bash /opt/apps/roboticsservice/runService.sh`
   (HandUMI also starts it automatically when recording).
2. Launch XRoboToolkit on the PICO and set the PC-service IP+port:
   `127.0.0.1:63901` for USB (default), or the workstation LAN IP for
   `--pico-wifi`.
3. Start streaming in the app (body tracking / controllers as needed).

### Smoke-test

```bash
handumi-record --device pico --skip-feetech \
  --repo-id local/pico_smoke --output-dir outputs/datasets/pico_smoke \
  --task "pico smoke" --num-episodes 1 --episode-time-s 10
```

Good = `xrobotoolkit_sdk initialised.` and no repeated
`still waiting for PICO data`. `--pico-mode mandos|object|whole-body`
selects what streams; `--manual-control` maps episodes to PICO buttons
(**A** start/stop, **B** repeat, **Y** finish).

### Troubleshooting

- **`xrt.init()` core dump** — PC service not running.
- **All zeros** — app not streaming, wrong IP/port, or (USB) missing
  `adb reverse --list` → `tcp:63901`.
- **SDK import error** — re-run `bash install.sh` without `--skip-xrt`.

---

Next: hardware + width calibration ([README_gripper.md](README_gripper.md)),
mount calibration ([README_calibration.md](README_calibration.md)), then
record per [README.md](../README.md).
