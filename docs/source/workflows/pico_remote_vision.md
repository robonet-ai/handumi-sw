# PICO Remote Vision

Stream the context and wrist cameras into a PICO headset independently of the
selected robot. This optional process does not import a robot backend, enable
CAN, or replace the normal real-teleoperation command.

## Protocol and preparation

HandUMI uses the XRoboToolkit Remote Vision protocol: commands travel through
port `13579` and length-prefixed H.264 video through port `12345`. Over USB the
directions are intentionally different:

```text
PICO command client -> adb reverse 13579 -> HandUMI bridge
HandUMI H.264 sender -> adb forward 12345 -> PICO decoder
```

The bridge uses the existing `ZEDMINI` source in the XRoboToolkit app and does
not install or replace `video_source.yml`. Close XRoboToolkit before creating
the tunnels because the running app normally owns device port `13579`. Also
close OBS or any viewer that owns a selected camera.

## Test one camera

Test video without enabling a robot:

```bash
uv run handumi camera pico \
  --camera /dev/video2 \
  --input-format mjpeg \
  --input-size 1280x720 \
  --fps 30 \
  --eye-y-offset 48
```

The default `--eye-y-offset 48` moves the complete image slightly downward in
both eyes. Use `0` for a centered image, a larger positive value to lower it
further, or a negative value to raise it.

When the terminal reports that Remote Vision is ready, open XRoboToolkit and
use:

```text
Remote Vision source: ZEDMINI
Camera source IP: 127.0.0.1
Listen: enabled
```

## Stream three cameras

One camera is fitted into each eye without stretching its aspect ratio. The
three-camera layout places the context camera in the center and the wrist
cameras at the sides:

```bash
uv run handumi camera pico \
  --camera /dev/video2 \
  --left-camera /dev/video4 \
  --right-camera /dev/video6
```

The process reads one frame from every configured camera before opening the
stream, so a missing, busy, or incompatible camera fails without involving the
robot. After validating video, leave this process running and start the normal
real-teleoperation command for the selected robot in another terminal.

If the video stream stops, teleoperation continues. The operator must decide
whether remote operation remains safe.

## Troubleshooting

If Remote Vision connects but stays blank, restart the bridge with the current
code and press `Listen` again. Check `adb logcat` for decoder errors.
XRoboToolkit requires 3-byte H.264 NALs to remain grouped inside the 4-byte
Annex-B decoder unit.
