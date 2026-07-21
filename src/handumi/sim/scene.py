"""Small scene primitives rendered by the Viser simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class SceneGeom:
    """A simple renderable geometry attached to a scene body frame."""

    kind: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    local_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    local_quaternion_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


@dataclass(frozen=True)
class SceneBody:
    """A named frame plus one or more local geometries for the Viser scene."""

    name: str
    geoms: tuple[SceneGeom, ...] = ()
    rest_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rest_quaternion_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


def _rgba4(text: str) -> tuple[float, float, float, float]:
    r, g, b, a = (float(v) for v in text.split())
    return (r, g, b, a)


SOURCE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = (
    SOURCE_ROOT / "assets" / "scenes"
    if (SOURCE_ROOT / "assets" / "scenes").exists()
    else PACKAGE_ROOT / "assets" / "scenes"
)

# Scene origin in the robot world: midpoint between arm bases, at base height.
DEFAULT_SCENE_POSITION = (0.30, 0.0, 0.0)


def load_scene(
    name: str, *, position: tuple[float, float, float] = DEFAULT_SCENE_POSITION
) -> list[SceneBody]:
    """Load ``assets/scenes/<name>/scene.xml`` (a plain MJCF fragment) into
    static :class:`SceneBody` primitives for the Viser renderer.

    Pure XML parsing — no MuJoCo. Only ``box`` geoms are supported; body
    ``pos`` offsets are added to the scene-level ``position`` (robot-world
    placement, default :data:`DEFAULT_SCENE_POSITION`).
    """
    path = SCENES_DIR / name / "scene.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No scene asset for {name!r}; expected {path}. "
            f"Add assets/scenes/{name}/scene.xml to define it."
        )
    offset = np.asarray(position, dtype=np.float32)
    root = ElementTree.parse(path).getroot()
    bodies: list[SceneBody] = []
    for body in root.iter("body"):
        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ", dtype=np.float32)
        geoms = []
        for geom in body.findall("geom"):
            if geom.get("type") != "box":
                continue
            geoms.append(
                SceneGeom(
                    kind="box",
                    size=tuple(float(v) for v in geom.get("size", "0.01").split()),
                    rgba=_rgba4(geom.get("rgba", "0.8 0.8 0.8 1")),
                    local_position=np.fromstring(
                        geom.get("pos", "0 0 0"), sep=" ", dtype=np.float32
                    ),
                )
            )
        if geoms:
            bodies.append(
                SceneBody(
                    name=body.get("name", f"body{len(bodies)}"),
                    geoms=tuple(geoms),
                    rest_position=offset + body_pos,
                )
            )
    return bodies


__all__ = ["DEFAULT_SCENE_POSITION", "SceneBody", "SceneGeom", "load_scene"]
