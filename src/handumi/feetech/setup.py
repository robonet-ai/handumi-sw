"""Interactive Feetech wiring setup helpers for ``configs/rig.yaml``."""

from __future__ import annotations

import glob
import getpass
import grp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from handumi.config import DEFAULT_RIG_CONFIG, EXAMPLE_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import default_config

DEFAULT_SERIAL_PATTERNS = ("/dev/ttyACM*", "/dev/ttyUSB*")
CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class FeetechServoRef:
    side: str
    port: str
    servo_id: int


def list_feetech_serial_ports(
    patterns: Iterable[str] = DEFAULT_SERIAL_PATTERNS,
) -> set[str]:
    """Return candidate USB serial ports for Feetech adapters."""
    ports: set[str] = set()
    for pattern in patterns:
        ports.update(glob.glob(pattern))
    return set(sorted(ports))


def scan_feetech_ids(
    port: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    bus_cls=FeetechBus,
) -> list[int]:
    if start_id > end_id:
        raise SystemExit("--feetech-start-id must be <= --feetech-end-id.")
    _assert_serial_port_access(port)
    with bus_cls(
        port=port,
        baudrate=baudrate,
        protocol_version=protocol_version,
    ) as bus:
        return [int(servo_id) for servo_id in bus.scan(range(start_id, end_id + 1))]


def identify_feetech_by_replug(
    side_label: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    scan_ids_fn: Callable[[str], list[int]] | None = None,
    used_ports: set[str] | None = None,
) -> FeetechServoRef:
    """Identify one Feetech adapter by disconnect/reconnect and scan its ID."""
    used_ports = set(used_ports or set())
    print_fn(f"\nIdentificando Feetech del HandUMI {side_label}.")
    input_fn(f"Desconecta el Feetech {side_label} y presiona Enter.")
    disconnected = list_ports_fn()
    input_fn(f"Conecta SOLO el Feetech {side_label} y presiona Enter.")
    print_fn("  Esperando que aparezca un puerto serial nuevo...")

    deadline = time.time() + timeout_s
    next_status_s = time.time() + 3.0
    last_seen: set[str] = set()
    while time.time() < deadline:
        current = list_ports_fn()
        last_seen = current
        added = sorted((current - disconnected) - used_ports)
        if len(added) == 1:
            return _scan_identified_feetech(
                side_label,
                added[0],
                start_id=start_id,
                end_id=end_id,
                baudrate=baudrate,
                protocol_version=protocol_version,
                scan_ids_fn=scan_ids_fn,
                print_fn=print_fn,
            )
        if len(added) > 1:
            raise SystemExit(
                "Se detectaron multiples puertos serial nuevos: "
                f"{', '.join(added)}. Conecta solo uno por paso."
            )
        if used_ports:
            existing_unused = sorted(current - used_ports)
            if len(existing_unused) == 1:
                print_fn(
                    "  No aparecio un puerto nuevo, pero hay un puerto Feetech "
                    "sin asignar; lo usare."
                )
                return _scan_identified_feetech(
                    side_label,
                    existing_unused[0],
                    start_id=start_id,
                    end_id=end_id,
                    baudrate=baudrate,
                    protocol_version=protocol_version,
                    scan_ids_fn=scan_ids_fn,
                    print_fn=print_fn,
                )
        now = time.time()
        if now >= next_status_s:
            print_fn(
                "  Aun no veo un puerto nuevo. "
                f"Puertos actuales: {_format_ports(current)}"
            )
            next_status_s = now + 3.0
        time.sleep(poll_s)
    raise SystemExit(
        f"No se detecto el Feetech {side_label} en {timeout_s:.0f}s.\n"
        f"Puertos antes: {_format_ports(disconnected)}\n"
        f"Puertos ahora: {_format_ports(last_seen)}\n"
        "Revisa que el adaptador haya enumerado como /dev/ttyACM* o /dev/ttyUSB*, "
        "que el cable sea de datos, y que conectaste solo ese Feetech despues de Enter."
    )


def _scan_identified_feetech(
    side_label: str,
    port: str,
    *,
    start_id: int,
    end_id: int,
    baudrate: int,
    protocol_version: int,
    scan_ids_fn: Callable[[str], list[int]] | None,
    print_fn: Callable[[str], None],
) -> FeetechServoRef:
    _assert_serial_port_access(port)
    scanner = scan_ids_fn or (
        lambda detected_port: scan_feetech_ids(
            detected_port,
            start_id=start_id,
            end_id=end_id,
            baudrate=baudrate,
            protocol_version=protocol_version,
        )
    )
    ids = scanner(port)
    if len(ids) == 1:
        ref = FeetechServoRef(side=side_label, port=port, servo_id=int(ids[0]))
        print_fn(f"  {side_label}: detectado port={ref.port}, servo_id={ref.servo_id}")
        return ref
    if not ids:
        raise SystemExit(
            f"{port} aparecio para {side_label}, pero ningun servo respondio. "
            "Revisa power, cableado, baudrate y rango de IDs."
        )
    raise SystemExit(
        f"{port} tiene multiples Feetech IDs {ids}. "
        "Conecta solo un servo o cambia IDs con handumi-set-servo-id."
    )


def _format_ports(ports: set[str]) -> str:
    return ", ".join(sorted(ports)) if ports else "ninguno"


def ensure_feetech_serial_permissions(
    *,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    runner: CommandRunner = subprocess.run,
    user: str | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Preflight serial permissions before the interactive setup starts."""
    ports = sorted(list_ports_fn())
    if not ports:
        return

    missing_groups: dict[str, list[str]] = {}
    blocked_ports: list[str] = []
    for port in ports:
        if os.access(port, os.R_OK | os.W_OK):
            continue
        group_name = _serial_port_group_name(port)
        if group_name and group_name not in _current_group_names():
            missing_groups.setdefault(group_name, []).append(port)
            continue
        blocked_ports.append(port)

    if missing_groups:
        target_user = user or getpass.getuser()
        details = "; ".join(
            f"{group}: {', '.join(group_ports)}"
            for group, group_ports in sorted(missing_groups.items())
        )
        print_fn(f"Feetech necesita permisos seriales ({details}).")
        print_fn("Necesito sudo para agregar tu usuario al grupo del puerto.")
        sudo = runner(["sudo", "-v"], check=False)
        if sudo.returncode != 0:
            raise SystemExit("No se pudo obtener sudo; no se cambiaron permisos seriales.")
        for group in sorted(missing_groups):
            result = runner(
                ["sudo", "usermod", "-aG", group, target_user],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise SystemExit(f"No se pudo agregar {target_user} a {group}.\n{stderr}")
        raise SystemExit(
            "Permisos seriales actualizados.\n"
            "Cierra sesion y vuelve a entrar, o reinicia la maquina, y corre de nuevo:\n"
            "  uv run handumi-setup-hardware --robot piper --device pico"
        )

    if blocked_ports:
        raise SystemExit(
            "No tengo permisos para abrir estos puertos Feetech: "
            f"{', '.join(blocked_ports)}.\n"
            "Tu usuario parece estar en el grupo correcto; revisa reglas udev "
            "o si otro proceso tiene abierto el puerto."
        )


def _assert_serial_port_access(port: str) -> None:
    if os.access(port, os.R_OK | os.W_OK):
        return
    hint = _serial_port_permission_hint(port)
    if hint:
        raise SystemExit(
            f"No tengo permisos para abrir {port}.\n"
            + "\n".join(hint)
        )
    raise SystemExit(f"No tengo permisos para abrir {port}.")


def _serial_port_permission_hint(port: str) -> list[str]:
    try:
        stat_result = os.stat(port)
    except OSError:
        return []
    group_name = _group_name_from_gid(stat_result.st_gid)

    if stat_result.st_gid in os.getgroups():
        return [
            "Tu usuario ya esta en el grupo del puerto, pero el acceso fallo.",
            "Revisa reglas udev o si otro proceso tiene abierto el puerto.",
        ]

    return [
        f"Agrega tu usuario al grupo `{group_name}`:",
        f"  sudo usermod -aG {group_name} $USER",
        "Luego cierra sesion y vuelve a entrar, o reinicia la maquina.",
    ]


def _serial_port_group_name(port: str) -> str:
    try:
        return _group_name_from_gid(os.stat(port).st_gid)
    except OSError:
        return ""


def _group_name_from_gid(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _current_group_names() -> set[str]:
    names: set[str] = set()
    for gid in os.getgroups():
        names.add(_group_name_from_gid(gid))
    return names


def save_feetech_mapping(
    *,
    rig_config: Path,
    left: FeetechServoRef,
    right: FeetechServoRef,
    baudrate: int,
    protocol_version: int,
) -> None:
    ensure_rig_config(rig_config)
    with rig_config.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    data["feetech"] = {
        "baudrate": int(baudrate),
        "protocol_version": int(protocol_version),
        "left": {"servo_id": int(left.servo_id), "port": left.port},
        "right": {"servo_id": int(right.servo_id), "port": right.port},
    }

    with rig_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_feetech_wizard(
    *,
    rig_config: Path = DEFAULT_RIG_CONFIG,
    start_id: int = 0,
    end_id: int = 20,
    baudrate: int | None = None,
    protocol_version: int | None = None,
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_ports_fn: Callable[[], set[str]] = list_feetech_serial_ports,
    scan_ids_fn: Callable[[str], list[int]] | None = None,
) -> tuple[FeetechServoRef, FeetechServoRef]:
    """Wizard that writes left/right Feetech port and ID assignments."""
    defaults = default_config()
    baudrate = int(baudrate if baudrate is not None else defaults.baudrate)
    protocol_version = int(
        protocol_version if protocol_version is not None else defaults.protocol_version
    )

    print_fn("Wizard Feetech: se mapeara derecha primero, luego izquierda.")
    right = identify_feetech_by_replug(
        "derecho",
        start_id=start_id,
        end_id=end_id,
        baudrate=baudrate,
        protocol_version=protocol_version,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
        list_ports_fn=list_ports_fn,
        scan_ids_fn=scan_ids_fn,
    )
    left = identify_feetech_by_replug(
        "izquierdo",
        start_id=start_id,
        end_id=end_id,
        baudrate=baudrate,
        protocol_version=protocol_version,
        timeout_s=timeout_s,
        poll_s=poll_s,
        input_fn=input_fn,
        print_fn=print_fn,
        list_ports_fn=list_ports_fn,
        scan_ids_fn=scan_ids_fn,
        used_ports={right.port},
    )
    if left.port == right.port:
        raise SystemExit(f"Feetech izquierdo y derecho usan el mismo puerto: {left.port}")

    save_feetech_mapping(
        rig_config=rig_config,
        left=left,
        right=right,
        baudrate=baudrate,
        protocol_version=protocol_version,
    )
    print_fn(
        f"Guardado en {rig_config}: "
        f"left={left.port}/id{left.servo_id}, right={right.port}/id{right.servo_id}"
    )
    return left, right


def ensure_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE_RIG_CONFIG, path)


__all__ = [
    "FeetechServoRef",
    "ensure_feetech_serial_permissions",
    "identify_feetech_by_replug",
    "list_feetech_serial_ports",
    "run_feetech_wizard",
    "save_feetech_mapping",
    "scan_feetech_ids",
]
