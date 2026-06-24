#!/usr/bin/env python3
"""
Lee un LeRobotDataset local y ejercita __getitem__, iteración y propiedades.

Uso
───
    uv run python test/read_dataset.py
    uv run python test/read_dataset.py --root datasets/dexumi_demo --repo-id local/dexumi_demo
    uv run python test/read_dataset.py --idx 0 --idx 10 --idx -1   # frames específicos
    uv run python test/read_dataset.py --episodes 0                 # sólo episodio 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dexumi.read_dataset")

# ── Rutas por defecto ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = _REPO_ROOT / "datasets" / "dexumi_demo"
DEFAULT_REPO_ID = "local/dexumi_demo"


# ── Helpers de impresión ───────────────────────────────────────────────────────

def _fmt_value(v: object) -> str:
    """Representación compacta de un tensor u otro valor."""
    if isinstance(v, torch.Tensor):
        if v.numel() <= 8:
            return f"Tensor{list(v.shape)} = {v.tolist()}"
        return f"Tensor{list(v.shape)} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}"
    return repr(v)


def print_frame(frame: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}── frame keys ({len(frame)}) ──")
    for key, val in frame.items():
        print(f"  {key:<45} {_fmt_value(val)}")


# ── Lógica principal ───────────────────────────────────────────────────────────

def run(
    root: Path,
    repo_id: str,
    indices: list[int],
    episodes: list[int] | None,
    delta_timestamps: dict[str, list[float]] | None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    log.info("Cargando dataset desde: %s  (repo_id=%s)", root, repo_id)

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )

    # ── Información general ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(ds)
    print(f"  fps            : {ds.fps}")
    print(f"  num_episodes   : {ds.num_episodes}")
    print(f"  num_frames     : {ds.num_frames}")
    print(f"  feature keys   : {list(ds.features.keys())}")
    print("=" * 60)

    # ── __getitem__ con índices explícitos ────────────────────────────────────
    for raw_idx in indices:
        # Normalizar índice negativo
        idx = raw_idx if raw_idx >= 0 else len(ds) + raw_idx
        if not (0 <= idx < len(ds)):
            log.warning("Índice %d fuera de rango [0, %d), saltando.", raw_idx, len(ds))
            continue

        log.info("Leyendo frame idx=%d …", idx)
        frame = ds[idx]
        print_frame(frame, label=f"idx={idx}")

    # ── Iteración completa (primeros N frames) ────────────────────────────────
    MAX_ITER = 5
    log.info("Iterando los primeros %d frames con DataLoader …", MAX_ITER)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # 0 = sin subprocesos, más fácil para depurar
    )

    for batch_idx, batch in enumerate(loader):
        print(f"\n── batch {batch_idx} ──")
        for key, val in batch.items():
            print(f"  {key:<45} {_fmt_value(val[0])}")  # primer elemento del batch
        if (batch_idx + 1) * loader.batch_size >= MAX_ITER:
            break

    # ── get_raw_item (sin decodificación de vídeo ni transforms) ─────────────
    log.info("get_raw_item(0) (Parquet puro, sin transforms) …")
    raw = ds.get_raw_item(0)
    print("\n── raw item [0] (sin decodificación) ──")
    for key, val in raw.items():
        print(f"  {key:<45} {_fmt_value(val)}")

    log.info("Listo.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Directorio raíz del dataset (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"repo_id lógico del dataset (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--idx",
        type=int,
        action="append",
        dest="indices",
        metavar="N",
        help="Índice(s) de frame a leer con __getitem__. Se puede repetir. Acepta negativos.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        metavar="E",
        help="Si se indica, sólo carga estos episodios (p. ej. --episodes 0 1).",
    )
    parser.add_argument(
        "--delta-timestamps",
        action="store_true",
        default=False,
        help=(
            "Activa delta_timestamps de ejemplo para observation.state y action "
            "(-1/fps, 0, +1/fps), mostrando ventanas temporales."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    indices: list[int] = args.indices if args.indices else [0, 1, -1]

    delta_timestamps = None
    if args.delta_timestamps:
        fps = 30  # se sobreescribirá si el dataset tiene otro fps
        dt = 1.0 / fps
        delta_timestamps = {
            "observation.state": [-dt, 0.0, dt],
            "action": [-dt, 0.0, dt],
        }
        log.info("delta_timestamps activados: %s", delta_timestamps)

    run(
        root=args.root,
        repo_id=args.repo_id,
        indices=indices,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
    )


if __name__ == "__main__":
    main()
