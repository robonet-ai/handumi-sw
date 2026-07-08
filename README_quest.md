# Meta Quest Tracking Setup (Phase 2)

One-time setup to stream Meta Quest controller/HMD poses to the workstation over
**TCP/JSON + UDP time-sync** (not WebXR). The Quest is **worn on the neck as a
tracking base** (no headset UI), a controller mounted on each gripper. Run
commands live in [README.md](README.md); design details in
[docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md).

Quest tracking does not use XRoboToolkit (that's PICO-only, see
[README_pico.md](README_pico.md)) — install with `bash install.sh --skip-xrt`
to skip fetching/building it entirely.

We reuse the prebuilt **YubiQuestApp** from
[yubi-sw](https://github.com/airoa-org/yubi-sw) (no custom headset app); it
streams poses in the exact format `handumi.tracking.meta_quest` parses.

## 0. Try it without a Quest (mock)

```bash
# terminal 1 — fake Quest (TCP + UDP sync on localhost)
python -m handumi.tracking.mock_quest_sender

# terminal 2 — live tracking to Rerun (no cameras/Feetech)
python -m handumi.capture.live_tracking_quest \
  --quest-ip 127.0.0.1 --skip-cameras --skip-feetech
```

Expect fps, a non-zero clock offset, and left/right trajectories in Rerun.

## 1. Enable Developer Mode (one-time)

1. Create/join a (free) org at <https://developers.meta.com/>.
2. **Meta Horizon** app → *Devices → your headset → Headset Settings → Developer
   Mode → on*.
3. Connect the headset to the laptop with **USB-C**.

## 2. Authorize the laptop over USB

`adb` talks to the Quest over USB (`sudo apt install adb` if missing).

```bash
adb devices
```

- `... unauthorized` → **put on the headset**, accept *Allow USB debugging*
  ("Always allow"). No prompt? `adb kill-server && adb start-server && adb devices`.
- `... device` → authorized (and Developer Mode is on).

## 3. Install the YubiQuestApp (one-time)

Downloaded into `external_dependencies/` (gitignored, same convention as the
PICO SDK build) — never into the repo root:

```bash
mkdir -p external_dependencies/quest-app
cd external_dependencies/quest-app
wget https://releases.dev.airoa.io/yubi/quest-app/yubi-quest-app-v0.1.0.apk
adb install -r yubi-quest-app-v0.1.0.apk        # -r upgrades in place
cd -
```

On `INSTALL_FAILED_UPDATE_INCOMPATIBLE` (different-signed build present):

```bash
adb uninstall com.UnityTechnologies.com.unity.template.urpblank
adb install -r external_dependencies/quest-app/yubi-quest-app-v0.1.0.apk
```

Appears in *Library → Unknown Sources → YubiQuestApp*. GUI alternative:
[SideQuest](https://sidequestvr.com/) drag-and-drop.

## 4. Find the Quest IP and set it once

Streaming is over Wi-Fi/LAN (USB is only for installing); the laptop dials the
Quest. **You can unplug the USB-C cable once the app is installed** — tracking
keeps working over Wi-Fi as long as both are on the same network. Put both on
the **same network**, then:

```bash
adb shell ip route          # prints e.g. ... src 10.104.18.172
ping -c 3 10.104.18.172      # 0% loss = reachable (guest Wi-Fi may isolate)
```

Pin it in `configs/tracking_meta_quest.yaml` (DHCP can change it):

```yaml
connection:
  quest_ip: "10.104.18.172"  # ← your Quest IP
  tcp_port: 65432            # YubiQuestApp defaults — leave as-is
  sync_port: 42000
```

## 5. Launch the app and smoke-test

The app **pauses when the headset isn't worn** (proximity sensor), so its TCP
server is up only while the app runs in the foreground **with the headset on**.

1. **Launch** YubiQuestApp in the headset (or from the laptop):

   ```bash
   adb shell monkey -p com.UnityTechnologies.com.unity.template.urpblank \
     -c android.intent.category.LAUNCHER 1
   ```

   It shows **`server running <ip>:<port>`** — the TCP pose server. Keep it foreground.
2. **Wake both controllers** and keep them in view of the headset cameras (off or
   occluded → `trk=0`, pose freezes).
3. **Check the port** (prints `OPEN`):

   ```bash
   timeout 5 bash -c 'exec 3<>/dev/tcp/10.104.18.172/65432 && echo OPEN || echo closed'
   ```
4. **Run the receiver** (`Ctrl+C` to stop) — one self-updating line:

   ```bash
   python -m handumi.tracking.meta_quest --config configs/tracking_meta_quest.yaml
   ```

   ```text
   seq=000000 fps=120.0 off=+188329.1156s rtt=7.19ms | L trk=1 [+0.11,-0.32,+0.55] R trk=1 [-0.09,-0.30,+0.55]
   ```

   Good = `fps` steady (~120), `off=` stable (only the last decimals wiggle), and
   **both `trk=1`** with positions that move as you move the controllers. `trk=0`
   → revisit step 2. (`seq` is always 0 — yubi's format carries none.)

## 6. See the 3D trajectory (Rerun)

```bash
python -m handumi.capture.live_tracking_quest --skip-cameras --skip-feetech
```

Move the controllers and their trails draw — **left cyan, right magenta**. A
**yellow marker** shows the workspace origin (the HMD pose captured at the last
reset) — both trails are positions *relative to that one fixed point*, not to
each other or to the live head position. The **left X button** re-centres the
workspace on the current HMD pose (also auto-set on the first tracked frame).

> Headless/SSH? Point at a remote Rerun viewer:
> `--display-ip <viewer-host> --display-port <port>` (or `--no-rerun-spawn`).

Add `--robot piper` to also render a Piper robot IK-following your hands in
Viser (http://localhost:8003; first launch JIT-compiles for ~30s).

Once this looks right, mount the headset on the neck and use the full
live/record commands (with cameras + Feetech) in [README.md](README.md).

## Troubleshooting

- **`adb` doesn't list the Quest** — replug USB-C, re-accept *Allow USB
  debugging*, `adb kill-server && adb devices`.
- **Port refused / no frames** — the app must be foreground with the headset
  worn; confirm `quest_ip` is current (DHCP) and both share the network (`ping`).
- **Poses frozen/zero** — controllers must be on and visible to the cameras.
- **Fields look wrong** — the APK defines the wire format. Dump a raw sample and
  compare against
  [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) → *TCP/JSON
  Payload*; adjust `handumi.tracking.meta_quest.parse_frame` if keys differ:

  ```bash
  python -m handumi.tracking.meta_quest \
    --config configs/tracking_meta_quest.yaml --print-raw
  ```
