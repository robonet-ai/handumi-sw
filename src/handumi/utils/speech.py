"""Cross-platform text-to-speech for recording feedback.

Same approach LeRobot uses (``lerobot.utils.utils.say``/``log_say``): shell out
to the OS's built-in speech synthesizer, so recorders can announce episode
state hands-free (the collector is wearing the HandUMI shells, not looking at
a terminal). No extra Python dependency.
"""

from __future__ import annotations

import logging
import platform
import subprocess

log = logging.getLogger("handumi.speech")


def say(text: str, *, blocking: bool = False) -> None:
    """Speak ``text`` using the OS's built-in TTS. No-op with a warning if
    the platform's speech command isn't available."""
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")
    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
    else:
        log.warning("Unsupported OS for text-to-speech (%s); skipping.", system)
        return

    try:
        if blocking:
            subprocess.run(cmd, check=True)
        else:
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0,
            )
    except FileNotFoundError:
        log.warning(
            "Text-to-speech command %r not found; install it (e.g. `spd-say` "
            "ships with speech-dispatcher on Linux) or pass --no-sounds.",
            cmd[0],
        )


def log_say(text: str, *, play_sounds: bool = True, blocking: bool = False) -> None:
    """Log ``text`` and, if ``play_sounds``, also speak it."""
    log.info(text)
    if play_sounds:
        say(text, blocking=blocking)
