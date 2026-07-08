# PICO / XRoboToolkit Tracking Setup

One-time setup to stream PICO body/controller poses to the workstation via
**XRoboToolkit** (PC service + headset app + Python SDK). Run recording commands
live in [README.md](README.md); PICO data lands in `observation.pico.*` fields.

`install.sh` (without `--skip-xrt`) already builds and installs `xrobotoolkit_sdk`;
you still need the **PC service** (workstation) and **PICO app** (headset) below.
If you only track with Meta Quest, use `bash install.sh --skip-xrt` instead —
see [README_quest.md](README_quest.md).

## 1. Install XRoboToolkit (one-time)

XRoboToolkit consists of a PC service (running on your workstation) and a PICO
app (running on the headset) that streams body-tracking data.

### PC Service

The PC service must be installed and running on your workstation before the
PICO can connect.

**Ubuntu 22.04 (x86_64 workstation):**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

**Ubuntu 24.04 (x86_64 workstation):**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

**Jetson (aarch64, onboard):**

```bash
sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb
```

See [XRoboToolkit-PC-Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)
for other platforms or newer versions.

After install the service script lives at
`/opt/apps/roboticsservice/runService.sh`. HandUMI starts it automatically when
recording; you can also launch it manually:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

### PICO App

1. Wear the PICO headset to begin the setup and installation process.
2. Complete the quick setup on PICO.
3. Make sure the PICO is connected to Wi-Fi.
4. Open the **browser** application in the PICO.
5. Type **xrobotoolkit** in the search bar and select the GitHub page
   **XR-Robotics**.
6. Make sure **Developer Mode** is enabled (*Settings → Developer*).
7. **[INSIDE PICO]** Scroll down in the GitHub page until you see the APK
   download option and click with the PICO trigger to download it.

   > Download **XRoboToolkit-PICO-1.1.1.apk** on PICO using the browser.
   > ([Other Versions](https://github.com/XR-Robotics/XRoboToolkit-PICO/releases))

8. **[INSIDE PICO]** Open the manage-downloads option on the top-right section
   of the browser page and click to open the
   **XRoboToolkit-PICO-1.1.1.apk** download.
9. **[INSIDE PICO]** Select **Install** — the application will appear in the
   *Unknown* section of your library.

## 2. Enable USB debugging (one-time, USB mode)

USB mode is the default HandUMI transport (`--pico-adb`). Connect the headset to
the workstation with **USB-C** and enable USB debugging on the PICO
(*Settings → Developer → USB debugging*).

`adb` talks to the PICO over USB (`sudo apt install adb` if missing):

```bash
adb devices
```

- `... unauthorized` → **put on the headset**, accept *Allow USB debugging*
  ("Always allow"). No prompt? `adb kill-server && adb start-server && adb devices`.
- `... device` → authorized.

## 3. Connect the PICO app to the PC service

Both sides must be on the same network for Wi-Fi mode; USB mode tunnels over
`adb reverse` so the PICO app can reach `127.0.0.1`.

1. **Start the PC service** (if not already running):

   ```bash
   bash /opt/apps/roboticsservice/runService.sh
   ```

2. **Launch XRoboToolkit** on the PICO (Library → Unknown → XRoboToolkit).
3. In the PICO app, set the **PC-service IP and port**:

   | Transport | HandUMI flag | PC-service IP in PICO app | Port |
   |-----------|--------------|---------------------------|------|
   | USB (default) | *(none — `--pico-adb` is default)* | `127.0.0.1` | `63901` |
   | Wi-Fi/LAN | `--pico-wifi` | your workstation LAN IP | `63901` |

   Find the workstation LAN IP:

   ```bash
   ip route get 8.8.8.8 | awk '{print $7; exit}'
   ```

4. **Start streaming** in the PICO app (body tracking / controllers as needed).

HandUMI sets up `adb reverse tcp:63901` automatically in USB mode and prints
the LAN IP hint in Wi-Fi mode when you run the recorder.

## 4. Smoke-test the pose stream

PICO-only smoke test (no wrist cameras or Feetech):

```bash
handumi-record-pico \
  --use-pico --only-pico --skip-feetech \
  --repo-id local/pico_smoke \
  --output-dir outputs/datasets/pico_smoke \
  --task "pico smoke test" \
  --num-episodes 1 \
  --episode-time-s 10 \
  --fps 30
```

Good signs in the log:

- `xrobotoolkit_sdk initialised.`
- `PICO body-tracking data is available.` (default whole-body mode)
- No repeated `still waiting for PICO data` after ~15 s

Move the controllers / body while recording; `observation.pico.*` fields in the
saved episode should be non-zero. Use `--pico-mandos` for controllers only, or
`--pico-object` for motion trackers.

## 5. Record a full HandUMI dataset

Once [README_gripper.md](README_gripper.md) camera + Feetech setup is done:

```bash
handumi-record-pico \
  --use-pico \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Add `--pico-wifi` if you are not using USB/ADB. Add `--manual-control` to
start/stop episodes from PICO buttons (**A** = start/stop, **B** = repeat,
**Y** = finish).

## Troubleshooting

- **`xrt.init()` crashes / core dump** — the PC service is not running. Install
  the `.deb` from step 1 and run
  `bash /opt/apps/roboticsservice/runService.sh`.
- **No ADB device found** — replug USB-C, enable USB debugging, accept the
  prompt on the headset, or switch to `--pico-wifi`.
- **PICO data not available / all zeros** — confirm the PICO app is streaming,
  the PC-service IP matches the table in step 3, and port `63901` is open. In
  USB mode, check `adb reverse --list` shows `tcp:63901`.
- **Body joints empty** — enable body tracking in the PICO app, or pass
  `--pico-mandos` if you only need controller poses.
- **SDK import error** — re-run `bash install.sh` (without `--skip-xrt`) to build
  `xrobotoolkit_sdk`.
