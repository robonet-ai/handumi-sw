# Meta Quest Tracking Setup (Phase 2)

One-time setup to stream Meta Quest controller/HMD poses to the workstation.
The Quest is **body-worn on the neck as a tracking base** (no headset UI) with a
controller mounted on each gripper. Poses arrive over **TCP/JSON + a UDP
time-sync** (not WebXR). Once set up, the run commands live in the main
[README.md](README.md). Design details:
[docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md).

## The Quest app

We do **not** build our own headset app — we reuse the prebuilt **YubiQuestApp**
from [yubi-sw](https://github.com/airoa-org/yubi-sw), which streams OVR
controller/HMD poses in the exact TCP/JSON format this repo's receiver parses.
(Building a dedicated HandUMI app is a possible future step.)

## 0. Try it now without a Quest (mock)

The whole Python pipeline runs against a built-in mock that emulates the Quest
app. Two terminals:

```bash
# terminal 1 — fake Quest (TCP pose stream + UDP time-sync on localhost)
python -m handumi.tracking.mock_quest_sender

# terminal 2 — live tracking to Rerun (no cameras/Feetech needed)
python -m handumi.capture.live_tracking \
  --quest-ip 127.0.0.1 --skip-cameras --skip-feetech
```

You should see fps, a non-zero clock offset, and the left/right controller
trajectories drawing in the Rerun 3D view.

## 1. Enable Developer Mode (one-time)

Sideloading needs Developer Mode, which needs a (free) Meta developer
organization:

1. Create/join an org at <https://developers.meta.com/> (one-time, on the web).
2. In the **Meta Horizon** mobile app (paired with the headset):
   *Menu → Devices → your headset → Headset Settings → Developer Mode → on*.
3. Connect the headset to the laptop with a **USB-C** cable.

## 2. Authorize the laptop over USB

`adb` talks to the Quest (Android) over USB — same `adb` the PICO setup uses
(`sudo apt install adb` if missing).

```bash
adb devices
```

- `3487C10H960BQZ   unauthorized` → **put on the headset** and accept the
  *Allow USB debugging* prompt (check "Always allow from this computer"). If the
  prompt doesn't show, run `adb kill-server && adb start-server && adb devices`.
- `3487C10H960BQZ   device` → authorized and Developer Mode is on. (If the
  device appears at all, Developer Mode is already enabled.)

## 3. Install the YubiQuestApp (one-time)

```bash
# a) download the prebuilt APK (or download it in a browser)
wget https://releases.dev.airoa.io/yubi/quest-app/yubi-quest-app-v0.1.0.apk

# b) install (-r upgrades over an existing install, keeping its data)
adb install -r yubi-quest-app-v0.1.0.apk
```

If it fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` (a different-signed build
is already there), uninstall it first, then reinstall:

```bash
adb uninstall com.UnityTechnologies.com.unity.template.urpblank
adb install -r yubi-quest-app-v0.1.0.apk
```

The app appears in the Quest library under **Unknown Sources** as
**YubiQuestApp** (package `com.UnityTechnologies.com.unity.template.urpblank`).
GUI alternative: [SideQuest](https://sidequestvr.com/), drag-and-drop the APK.

## 4. Find the Quest IP and set it once

The USB cable is only for installing — streaming is over **Wi-Fi/LAN**, and the
laptop dials the Quest. Put both on the **same network**, then find the Quest IP:

```bash
adb shell ip route          # prints e.g. ... src 10.104.18.172
# or on the headset: Settings → Wi-Fi → (your network) → details
```

Confirm the laptop can reach it (campus/guest Wi-Fi sometimes isolates clients,
which would block streaming):

```bash
ping -c 3 10.104.18.172     # 0% packet loss = reachable
```

Set the IP once in `configs/tracking_meta_quest.yaml` so you don't pass
`--quest-ip` every time:

```yaml
connection:
  quest_ip: "10.104.18.172"  # ← your Quest IP (DHCP can change it)
  tcp_port: 65432            # YubiQuestApp defaults — leave as-is
  sync_port: 42000
```

## 5. Launch the app and smoke-test

The app **pauses when the headset isn't worn** (proximity sensor), so its TCP
server only comes up while the app runs in the foreground **with the headset on**.

**Step 1 — launch YubiQuestApp.** Put on the headset and open it from *Library →
Unknown Sources → YubiQuestApp*, or kick it from the laptop:

```bash
adb shell monkey -p com.UnityTechnologies.com.unity.template.urpblank \
  -c android.intent.category.LAUNCHER 1
```

Inside the headset the app shows **`server running <ip>:<port>`** (e.g.
`10.104.18.172:65432`) — that is the TCP pose server. Keep the app in the
foreground.

**Step 2 — turn on the controllers.** Power on **both** controllers (press a
button/joystick to wake them) and keep them **in view of the headset cameras**.
When occluded or off, they report `tracked=0` and their pose freezes.

**Step 3 — confirm the port is open** (should print `ABIERTO`):

```bash
timeout 5 bash -c 'exec 3<>/dev/tcp/10.104.18.172/65432 && echo ABIERTO || echo cerrado'
```

**Step 4 — run the receiver** (simplest "is data flowing?" check, no Rerun, no
cameras). It prints one self-updating line; `Ctrl+C` to stop:

```bash
python -m handumi.tracking.meta_quest \
  --config configs/tracking_meta_quest.yaml
```

Read the line like this:

```text
seq=000000 fps=120.0 off=+188329.1156s rtt=7.19ms | L trk=1 [+0.11,-0.32,+0.55] R trk=1 [-0.09,-0.30,+0.55]
        │        │            │              │           │        └─ position moves as you move the controller
        │        │            │              │           └─ trk=1 means tracked (trk=0 = off/occluded, pose frozen)
        │        │            │              └─ round-trip time of the UDP sync (a few ms)
        │        │            └─ clock offset: large but must be STABLE (only the last decimals wiggle)
        │        └─ frame rate (~120 is healthy)
        └─ always 0 — yubi's legacy format carries no seq
```

Success = `fps` steady (~120), `off=` stable, and **both `trk=1` with positions
that change as you move the controllers**. If `trk=0`, revisit Step 2.

## 6. See the 3D trajectory (Rerun)

With the stream verified, visualize the controllers as a live 3D trajectory —
still without the gripper hardware, so skip cameras and Feetech:

```bash
python -m handumi.capture.live_tracking --skip-cameras --skip-feetech
```

A Rerun window opens with a 3D grid. Move the controllers and their trajectories
draw — **left cyan, right magenta** (like the yubi trail). The **left X button**
re-centres the workspace on the current HMD pose (also auto-set on the first
tracked frame).

> Headless / over SSH (no window)? Start a Rerun viewer where you can see it and
> point the script at it over gRPC:
> `... --skip-cameras --skip-feetech --display-ip <viewer-host> --display-port <port>`
> (or `--no-rerun-spawn` to run without any viewer).

Once this looks right, the headset can go on the neck mount and you're ready for
the full live/record commands (with cameras + Feetech) in [README.md](README.md).

## Troubleshooting

- **`adb` doesn't list the Quest** — replug the USB-C cable, re-accept *Allow USB
  debugging* in the headset, try `adb kill-server && adb devices`.
- **TCP port refused / receiver connecting but no frames** — YubiQuestApp must be
  **running in the foreground with the headset worn** (it sleeps otherwise);
  check `quest_ip` is the headset's current Wi-Fi IP (DHCP can change it) and
  that laptop + Quest share the network (`ping` above).
- **Connects but poses are frozen/zero** — the controllers must be on and visible
  to the headset cameras; `tracked`/`valid` go false when occluded.
- **Frames arrive but fields look wrong** — the APK is the source of truth for the
  wire format. Dump one raw sample and compare it against
  [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) →
  *TCP/JSON Payload*:

  ```bash
  python -m handumi.tracking.meta_quest \
    --config configs/tracking_meta_quest.yaml --print-raw
  ```

  If key names differ, adjust `handumi.tracking.meta_quest.parse_frame`.
