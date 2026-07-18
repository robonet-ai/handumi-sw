"""CAN interface discovery, repair, and rig.yaml setup helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from handumi.config import DEFAULT_RIG_CONFIG, EXAMPLE_RIG_CONFIG

CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class CanInterfaceStatus:
    name: str
    exists: bool
    flags: tuple[str, ...] = ()
    state: str | None = None
    bitrate: int | None = None
    dbitrate: int | None = None
    fd: bool = False

    @property
    def up(self) -> bool:
        return "UP" in self.flags

    @property
    def lower_up(self) -> bool:
        return "LOWER_UP" in self.flags

    @property
    def bus_off(self) -> bool:
        return self.state == "BUS-OFF"


@dataclass(frozen=True)
class CanInterfaceRef:
    name: str
    sysfs_path: str


def list_can_interfaces(
    sys_class_net: Path = Path("/sys/class/net"),
) -> dict[str, CanInterfaceRef]:
    """Return current ``can*`` net devices keyed by interface name."""
    if not sys_class_net.exists():
        return {}
    refs: dict[str, CanInterfaceRef] = {}
    for entry in sorted(sys_class_net.iterdir()):
        if not entry.name.startswith("can"):
            continue
        refs[entry.name] = CanInterfaceRef(
            name=entry.name,
            sysfs_path=str(entry.resolve()),
        )
    return refs


def read_can_status(
    name: str,
    *,
    runner: CommandRunner = subprocess.run,
) -> CanInterfaceStatus:
    result = runner(
        ["ip", "-details", "link", "show", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return CanInterfaceStatus(name=name, exists=False)

    stdout = result.stdout or ""
    flags: tuple[str, ...] = ()
    flag_match = re.search(rf"\d+:\s+{re.escape(name)}:\s+<([^>]*)>", stdout)
    if flag_match:
        flags = tuple(
            item.strip() for item in flag_match.group(1).split(",") if item.strip()
        )

    state_match = re.search(r"\bcan state\s+([A-Z-]+)", stdout)
    bitrate_match = re.search(r"\bbitrate\s+(\d+)", stdout)
    dbitrate_match = re.search(r"\bdbitrate\s+(\d+)", stdout)
    return CanInterfaceStatus(
        name=name,
        exists=True,
        flags=flags,
        state=state_match.group(1) if state_match else None,
        bitrate=int(bitrate_match.group(1)) if bitrate_match else None,
        dbitrate=int(dbitrate_match.group(1)) if dbitrate_match else None,
        fd="<FD>" in stdout or bool(dbitrate_match),
    )


def can_ready(status: CanInterfaceStatus, *, bitrate: int) -> bool:
    return (
        status.exists
        and status.up
        and status.lower_up
        and status.bitrate == int(bitrate)
        and not status.bus_off
    )


def can_status_reason(status: CanInterfaceStatus, *, bitrate: int) -> str:
    if not status.exists:
        return "missing"
    reasons = []
    if not status.up:
        reasons.append("down")
    if not status.lower_up:
        reasons.append("no cable/link")
    if status.bitrate != int(bitrate):
        reasons.append(f"bitrate={status.bitrate}")
    if status.bus_off:
        reasons.append("BUS-OFF")
    if not reasons:
        return "ready"
    return ", ".join(reasons)


def ensure_can_interfaces_ready(
    ports: list[str] | tuple[str, ...],
    *,
    bitrate: int,
    restart_ms: int,
    repair: bool = True,
    runner: CommandRunner = subprocess.run,
    print_fn: Callable[[str], None] = print,
) -> dict[str, CanInterfaceStatus]:
    """Repair CAN interfaces with explicit sudo, then return final statuses."""
    unique_ports = list(dict.fromkeys(ports))
    if len(unique_ports) != len(ports):
        raise SystemExit(f"Duplicate CAN ports configured: {', '.join(ports)}")

    statuses = {port: read_can_status(port, runner=runner) for port in unique_ports}
    if all(can_ready(status, bitrate=bitrate) for status in statuses.values()):
        return statuses

    if not repair:
        details = "; ".join(
            f"{port}: {can_status_reason(status, bitrate=bitrate)}"
            for port, status in statuses.items()
        )
        raise SystemExit(f"CAN is not ready ({details}); no robot will move.")

    details = "; ".join(
        f"{port}: {can_status_reason(status, bitrate=bitrate)}"
        for port, status in statuses.items()
    )
    print_fn(f"CAN necesita reparacion ({details}).")
    print_fn("Necesito sudo para activar CAN. Escribe tu password si aceptas.")
    sudo = runner(["sudo", "-v"], check=False)
    if sudo.returncode != 0:
        raise SystemExit("No se pudo obtener sudo; no se movera el robot.")

    for port, status in statuses.items():
        if can_ready(status, bitrate=bitrate):
            continue
        repair_can_interface(
            port,
            bitrate=bitrate,
            restart_ms=restart_ms,
            runner=runner,
        )

    final = {port: read_can_status(port, runner=runner) for port in unique_ports}
    not_ready = {
        port: status
        for port, status in final.items()
        if not can_ready(status, bitrate=bitrate)
    }
    if not_ready:
        details = "; ".join(
            f"{port}: {can_status_reason(status, bitrate=bitrate)}"
            for port, status in not_ready.items()
        )
        raise SystemExit(f"CAN no quedo listo ({details}); no se movera el robot.")
    return final


def repair_can_interface(
    port: str,
    *,
    bitrate: int,
    restart_ms: int,
    runner: CommandRunner = subprocess.run,
) -> None:
    commands = [
        ["sudo", "ip", "link", "set", port, "down"],
        [
            "sudo",
            "ip",
            "link",
            "set",
            port,
            "type",
            "can",
            "bitrate",
            str(int(bitrate)),
            "restart-ms",
            str(int(restart_ms)),
        ],
        ["sudo", "ip", "link", "set", port, "up"],
    ]
    for command in commands:
        result = runner(command, capture_output=True, text=True, check=False)
        if (
            result.returncode != 0
            and "restart-ms" in command
            and _restart_ms_unsupported(result.stderr or "")
        ):
            fallback = [
                "sudo",
                "ip",
                "link",
                "set",
                port,
                "type",
                "can",
                "bitrate",
                str(int(bitrate)),
            ]
            result = runner(fallback, capture_output=True, text=True, check=False)
        if result.returncode != 0 and command[-1] != "down":
            stderr = (result.stderr or "").strip()
            raise SystemExit(f"Fallo reparando {port}: {' '.join(command)}\n{stderr}")


def _restart_ms_unsupported(stderr: str) -> bool:
    text = stderr.lower()
    return "restart" in text and (
        "doesn't support" in text
        or "does not support" in text
        or "operation not supported" in text
        or "not supported" in text
    )


def ensure_can_fd_interfaces_ready(
    ports: list[str] | tuple[str, ...],
    *,
    bitrate: int,
    dbitrate: int,
    repair: bool = True,
    runner: CommandRunner = subprocess.run,
    print_fn: Callable[[str], None] = print,
) -> dict[str, CanInterfaceStatus]:
    """Validate OpenArm CAN-FD links and optionally configure them with sudo."""
    unique_ports = list(dict.fromkeys(ports))
    if len(unique_ports) != len(ports):
        raise SystemExit(f"Duplicate CAN ports configured: {', '.join(ports)}")

    def ready(status: CanInterfaceStatus) -> bool:
        return (
            can_ready(status, bitrate=bitrate)
            and status.fd
            and status.dbitrate == int(dbitrate)
        )

    statuses = {port: read_can_status(port, runner=runner) for port in unique_ports}
    if all(ready(status) for status in statuses.values()):
        return statuses
    if not repair:
        details = "; ".join(
            f"{port}: {can_status_reason(status, bitrate=bitrate)}, "
            f"fd={status.fd}, dbitrate={status.dbitrate}"
            for port, status in statuses.items()
        )
        raise SystemExit(f"OpenArm CAN-FD is not ready ({details}).")

    print_fn("OpenArm CAN-FD needs configuration; sudo is required.")
    if runner(["sudo", "-v"], check=False).returncode != 0:
        raise SystemExit("No se pudo obtener sudo; no se movera el robot.")
    for port, status in statuses.items():
        if ready(status):
            continue
        repair_can_fd_interface(port, bitrate=bitrate, dbitrate=dbitrate, runner=runner)
    final = {port: read_can_status(port, runner=runner) for port in unique_ports}
    if not all(ready(status) for status in final.values()):
        raise SystemExit("OpenArm CAN-FD did not become ready; no robot will move.")
    return final


def repair_can_fd_interface(
    port: str,
    *,
    bitrate: int,
    dbitrate: int,
    runner: CommandRunner = subprocess.run,
) -> None:
    commands = (
        ["sudo", "ip", "link", "set", port, "down"],
        [
            "sudo",
            "ip",
            "link",
            "set",
            port,
            "type",
            "can",
            "bitrate",
            str(int(bitrate)),
            "dbitrate",
            str(int(dbitrate)),
            "fd",
            "on",
        ],
        ["sudo", "ip", "link", "set", port, "up"],
    )
    for command in commands:
        result = runner(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 and command[-1] != "down":
            raise SystemExit(
                f"Failed to configure OpenArm CAN-FD {port}: "
                f"{(result.stderr or '').strip()}"
            )


def ensure_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE_RIG_CONFIG, path)


def save_piper_can_mapping(
    *,
    rig_config: Path,
    left_port: str,
    right_port: str,
    bitrate: int,
    restart_ms: int,
    left_usb_path: str = "",
    right_usb_path: str = "",
) -> None:
    ensure_rig_config(rig_config)
    with rig_config.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    robots = data.setdefault("robots", {})
    piper = robots.setdefault("piper", {})
    piper["can"] = {
        "bitrate": int(bitrate),
        "restart_ms": int(restart_ms),
        "left_port": left_port,
        "right_port": right_port,
    }
    if left_usb_path or right_usb_path:
        piper["can"]["left_usb_path"] = left_usb_path
        piper["can"]["right_usb_path"] = right_usb_path
    with rig_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def identify_can_by_replug(
    side_label: str,
    *,
    sys_class_net: Path = Path("/sys/class/net"),
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> CanInterfaceRef:
    print_fn(f"\nIdentificando CAN del brazo {side_label}.")
    input_fn(f"Desconecta el CAN del brazo {side_label} y presiona Enter.")
    disconnected = list_can_interfaces(sys_class_net)
    input_fn(f"Conecta SOLO el CAN del brazo {side_label} y presiona Enter.")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        current = list_can_interfaces(sys_class_net)
        added = [ref for name, ref in current.items() if name not in disconnected]
        if len(added) == 1:
            print_fn(f"  {side_label}: detectado {added[0].name}")
            return added[0]
        if len(added) > 1:
            names = ", ".join(ref.name for ref in added)
            raise SystemExit(f"Se detectaron multiples CAN nuevos: {names}")
        time.sleep(poll_s)
    raise SystemExit(
        f"No se detecto el CAN del brazo {side_label} en {timeout_s:.0f}s."
    )


def run_piper_can_wizard(
    *,
    rig_config: Path,
    bitrate: int,
    restart_ms: int,
    sys_class_net: Path = Path("/sys/class/net"),
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> tuple[str, str]:
    print_fn("Wizard CAN Piper: se mapeara derecha primero, luego izquierda.")
    right = identify_can_by_replug(
        "derecho",
        sys_class_net=sys_class_net,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
    )
    left = identify_can_by_replug(
        "izquierdo",
        sys_class_net=sys_class_net,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
    )
    save_piper_can_mapping(
        rig_config=rig_config,
        left_port=left.name,
        right_port=right.name,
        bitrate=bitrate,
        restart_ms=restart_ms,
        left_usb_path=left.sysfs_path,
        right_usb_path=right.sysfs_path,
    )
    print_fn(f"Guardado en {rig_config}: left={left.name}, right={right.name}")
    return left.name, right.name


def save_openarm_can_mapping(
    *,
    rig_config: Path,
    left_port: str,
    right_port: str,
    bitrate: int,
    dbitrate: int,
    left_usb_path: str = "",
    right_usb_path: str = "",
) -> None:
    ensure_rig_config(rig_config)
    with rig_config.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    can = (
        data.setdefault("robots", {}).setdefault("openarmv1", {}).setdefault("can", {})
    )
    can.update(
        {
            "fd": True,
            "bitrate": int(bitrate),
            "dbitrate": int(dbitrate),
            "left_port": left_port,
            "right_port": right_port,
        }
    )
    if left_usb_path or right_usb_path:
        can["left_usb_path"] = left_usb_path
        can["right_usb_path"] = right_usb_path
    with rig_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_openarm_can_wizard(
    *,
    rig_config: Path,
    bitrate: int = 1_000_000,
    dbitrate: int = 5_000_000,
    sys_class_net: Path = Path("/sys/class/net"),
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> tuple[str, str]:
    del timeout_s, poll_s  # OpenArm uses explicit can0/can1 selection, not replugging.
    interfaces = list_can_interfaces(sys_class_net)
    if len(interfaces) < 2:
        detected = ", ".join(interfaces) or "ninguna"
        raise SystemExit(
            "OpenArm v1 requires two CAN interfaces. "
            f"Detected: {detected}. Connect both adapters and retry."
        )
    choices = ", ".join(interfaces)
    print_fn(f"OpenArm CAN interfaces detected: {choices}")
    print_fn(
        "Indica el lado fisico de cada interfaz; no es necesario desconectar cables."
    )
    right_name = input_fn(f"CAN del brazo derecho ({choices}): ").strip()
    left_name = input_fn(f"CAN del brazo izquierdo ({choices}): ").strip()
    for side, name in (("derecho", right_name), ("izquierdo", left_name)):
        if name not in interfaces:
            raise SystemExit(
                f"CAN del brazo {side} invalido: {name or '<vacio>'}. "
                f"Opciones: {choices}."
            )
    if right_name == left_name:
        raise SystemExit("OpenArm left and right must use different CAN interfaces.")
    right = interfaces[right_name]
    left = interfaces[left_name]
    save_openarm_can_mapping(
        rig_config=rig_config,
        left_port=left.name,
        right_port=right.name,
        bitrate=bitrate,
        dbitrate=dbitrate,
        left_usb_path=left.sysfs_path,
        right_usb_path=right.sysfs_path,
    )
    print_fn(
        f"Saved OpenArm CAN mapping in {rig_config}: "
        f"right={right.name}, left={left.name}."
    )
    return left.name, right.name


__all__ = [
    "CanInterfaceRef",
    "CanInterfaceStatus",
    "can_ready",
    "can_status_reason",
    "ensure_can_interfaces_ready",
    "ensure_can_fd_interfaces_ready",
    "ensure_rig_config",
    "identify_can_by_replug",
    "list_can_interfaces",
    "read_can_status",
    "repair_can_interface",
    "run_piper_can_wizard",
    "run_openarm_can_wizard",
    "save_openarm_can_mapping",
    "save_piper_can_mapping",
]
