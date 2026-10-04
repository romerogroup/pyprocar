# van Alphen orbits: the box and plane slicer text against the tiled marching-cubes reference.
# Fixture: data/examples/fermi3d/van-alphen (fcc Au, 15^3 Gamma grid, IBZ EIGENVAL)
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from references.orbits import (
    SHELL,
    BandGrid,
    OrbitCut,
    Plane,
    cell_surface,
    plane_orbits,
    read_vasp_grid,
    validate,
)
from references.validation import require
from scipy.constants import elementary_charge, hbar
from scipy.spatial import HalfspaceIntersection
from verify_steps import CALC, EV, finish, png, step

from pyprocar.core.fermisurface import FermiSurface
from pyprocar.plotter.fs_plot import FermiPlotter

TWO_PI_SQUARED = (2 * np.pi) ** 2
AREA_TOLERANCE = 1e-4
"""Ang^-2, 2 pi included. The text rounds to 4 decimals (5e-5)."""
FREQUENCY_TOLERANCE = 1e-5
"""Relative. The frequency text keeps every digit of the largest area. pyprocar's energies are
5e-7 eV off EIGENVAL's, which moves the mesh vertices: measured 1.5e-6."""
SNAP_ANGLE = 3e-4
"""Radians, the documented snap_normal rule: a normal this close to a real-space lattice
direction [u v w] with |indices| <= 4 becomes that direction."""
LATTICE_NOISE = 1e-6
"""Radians. pyprocar's reciprocal lattice is 4.6e-10 off POSCAR's, so an exact lattice direction
built from POSCAR may or may not get a snap note; above this angle the note is required."""
N_RANDOM = 20
SEED = 20261004


@dataclass
class _Loaded:
    grid: BandGrid | None = None
    surfaces: list[pv.PolyData] = field(default_factory=list)
    fs: FermiSurface | None = None


LOADED = _Loaded()


@dataclass(frozen=True)
class Cut:
    name: str
    normal: np.ndarray
    origin: np.ndarray


@step("analytic_validation")
def _():
    return validate()


@step("fixture_grid")
def _():
    grid = read_vasp_grid(CALC)
    bands = [b for b, e in enumerate(grid.energies) if e.min() < grid.fermi < e.max()]
    surfaces = [cell_surface(grid.energies[b], grid.fermi, grid.lattice) for b in bands]
    LOADED.grid = grid
    LOADED.surfaces = [s for s in surfaces if s is not None]
    require(grid.n_operations == 48, f"{grid.n_operations} lattice operations, not 48 (Oh)")
    require(grid.image_spread < 1e-6, f"coinciding images differ by {grid.image_spread} eV")
    require(
        grid.energies.shape == (20, 15, 15, 15),
        f"grid shape {grid.energies.shape}, not (20, 15, 15, 15)",
    )
    require(
        bands == [5] and len(LOADED.surfaces) == 1,
        f"bands {bands} with {len(LOADED.surfaces)} surfaces, not [5] with 1",
    )
    return {
        "fermi_outcar": grid.fermi,
        "n_operations": grid.n_operations,
        "image_spread_ev": grid.image_spread,
        "grid": list(grid.energies.shape[1:]),
        "bands_crossing_fermi": bands,
        "n_triangles": int(LOADED.surfaces[0].n_cells),
    }


@step("pyprocar_surface_matches_raw_grid")
def _():
    grid = LOADED.grid
    assert grid is not None
    fs = FermiSurface.from_code(code="vasp", dirpath=CALC, fermi=grid.fermi)
    LOADED.fs = fs
    ebs = fs.original_ebs
    n = np.array(grid.energies.shape[1:])
    index = np.rint(np.asarray(ebs.kpoints) * n).astype(int) % n
    bands = ebs.get_property("bands")
    assert bands is not None
    theirs = np.asarray(bands.value)[:, :, 0]
    mine = grid.energies[:, index[:, 0], index[:, 1], index[:, 2]].T
    lattice_diff = float(np.abs(np.asarray(fs.reciprocal_lattice) - grid.lattice).max())
    energy_diff = float(np.abs(theirs - mine).max())
    keys = sorted([int(b), int(s)] for b, s in fs.band_isosurfaces)
    require(lattice_diff < 1e-6, f"reciprocal lattice differs by {lattice_diff}")
    require(energy_diff < 1e-6, f"unfolded energies differ by {energy_diff} eV")
    require(
        len(index) == n.prod() and abs(fs.isovalue - grid.fermi) < 1e-12,
        f"{len(index)} grid points, isovalue {fs.isovalue} vs E_F {grid.fermi}",
    )
    require(keys == [[5, 0]], f"band isosurfaces {keys}, not [[5, 0]]")
    return {
        "reciprocal_lattice_max_diff": lattice_diff,
        "grid_energy_max_diff_ev": energy_diff,
        "n_kpoints": len(index),
        "isovalue": fs.isovalue,
        "band_isosurfaces": keys,
    }


def _directions(real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.array(
        [
            uvw
            for uvw in itertools.product(range(-4, 5), repeat=3)
            if any(uvw) and np.gcd.reduce(np.abs(uvw)) == 1
        ]
    )
    vectors = indices @ real
    return indices, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _snapped(normal: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, list[int] | None, float]:
    """The reference normal under the snap rule, the direction indices when it is within
    SNAP_ANGLE of one (exact directions included), and the angle to the nearest one."""
    unit = normal / np.linalg.norm(normal)
    indices, directions = _directions(real)
    best = int(np.argmax(directions @ unit))
    angle = float(
        np.arctan2(np.linalg.norm(np.cross(directions[best], unit)), directions[best] @ unit)
    )
    if angle <= SNAP_ANGLE:
        return directions[best], [int(i) for i in indices[best]], angle
    return unit, None, angle


def _cuts(grid: BandGrid, points: np.ndarray) -> list[Cut]:
    real = grid.real_lattice
    metric = real @ real.T
    require(np.allclose(metric, metric[0, 0] * (np.eye(3) + 1) / 2), "a1, a2, a3 are not fcc")
    a1, a2, a3 = real
    gamma = np.zeros(3)
    cuts = [
        Cut("cubic001_gamma", a1 + a2 - a3, gamma),
        Cut("cubic111_gamma", a1 + a2 + a3, gamma),
        Cut("cubic110_gamma", a3, gamma),
        Cut("cubic111_L", a1 + a2 + a3, 0.5 * grid.lattice.sum(axis=0)),
        Cut(
            "cubic111_gamma_typed_4_digits",
            np.round((a1 + a2 + a3) / np.linalg.norm(a1 + a2 + a3), 4),
            gamma,
        ),
    ]
    rng = np.random.default_rng(SEED)
    for i in range(N_RANDOM):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        heights = points @ normal
        cuts.append(
            Cut(f"random{i:02d}", normal, rng.uniform(heights.min(), heights.max()) * normal)
        )
    return cuts


def _parse_area(text: str) -> tuple[float, int, list[int] | None]:
    area = re.search(r"Cross sectional area : ([0-9.]+) Ang\^-2", text)
    assert area is not None, text
    n_open = re.search(r"\((\d+) open curves? not counted\)", text)
    snap = re.search(r"normal snapped to \[(-?\d+) (-?\d+) (-?\d+)\]", text)
    return (
        float(area.group(1)),
        int(n_open.group(1)) if n_open else 0,
        [int(i) for i in snap.groups()] if snap else None,
    )


def _parse_frequency(text: str) -> float | None:
    found = re.search(r"Van Alphen Frequency : ([0-9.]+) Gauss", text)
    if found is None:
        require("no closed orbit through this cut" in text, text)
        return None
    return float(found.group(1))


def _onsager_gauss(area: float) -> float:
    return hbar * area * TWO_PI_SQUARED * 1e20 / (2 * np.pi * elementary_charge) * 1e4


def _widget_text(fs: FermiSurface, cut: Cut, box: bool) -> str:
    p = FermiPlotter(off_screen=True)
    if box:
        p.add_box_slicer(
            fs, normal=tuple(cut.normal), origin=tuple(cut.origin), show_cross_section_area=True
        )
    else:
        p.add_slicer(
            fs, normal=tuple(cut.normal), origin=tuple(cut.origin), show_van_alphen_frequency=True
        )
    text = cast(pv.CornerAnnotation, p.actors["area_text"]).GetText(2)
    p.close()
    return text


REFERENCE_CUTS: dict[str, tuple[Cut, np.ndarray, OrbitCut]] = {}


@step("widget_cuts_vs_reference")
def _():
    grid, fs = LOADED.grid, LOADED.fs
    assert grid is not None and fs is not None
    records, mismatches = [], []
    for cut in _cuts(grid, np.asarray(fs.points)):
        normal, ref_snap, angle = _snapped(cut.normal, grid.real_lattice)
        ref = plane_orbits(LOADED.surfaces, grid.lattice, Plane(normal, cut.origin))
        REFERENCE_CUTS[cut.name] = (cut, normal, ref)
        area_text = _widget_text(fs, cut, box=True)
        frequency_text = _widget_text(fs, cut, box=False)
        area, n_open, snap = _parse_area(area_text)
        frequency = _parse_frequency(frequency_text)
        ref_area = sum(ref.areas) * TWO_PI_SQUARED
        ref_frequency = _onsager_gauss(max(ref.areas)) if ref.areas else None
        if ref_snap is None:
            snap_ok = snap is None
        elif angle > LATTICE_NOISE:
            snap_ok = snap == ref_snap
        else:
            snap_ok = snap in (None, ref_snap)
        frequency_ok = (frequency is None and ref_frequency is None) or (
            frequency is not None
            and ref_frequency is not None
            and abs(frequency - ref_frequency) <= FREQUENCY_TOLERANCE * ref_frequency
        )
        record = {
            "cut": cut.name,
            "normal": cut.normal.tolist(),
            "origin": cut.origin.tolist(),
            "angle_to_lattice_direction": angle,
            "reference_direction": ref_snap,
            "pyprocar_snap_note": snap,
            "box_text": area_text,
            "slicer_text": frequency_text,
            "pyprocar_area": area,
            "reference_area": ref_area,
            "reference_orbits": [a * TWO_PI_SQUARED for a in ref.areas],
            "pyprocar_open": n_open,
            "reference_open": ref.n_open,
            "reference_open_at_edge": ref.open_at_edge,
            "reference_reach": ref.reach,
            "reference_defects": ref.n_defects,
            "pyprocar_frequency_gauss": frequency,
            "reference_frequency_gauss": ref_frequency,
        }
        records.append(record)
        if (
            abs(area - ref_area) > AREA_TOLERANCE
            or n_open != ref.n_open
            or not snap_ok
            or not frequency_ok
            or ref.n_defects
        ):
            mismatches.append(cut.name)
    (EV / "cuts.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    require(not mismatches, f"cuts that differ (see cuts.json): {mismatches}")
    typed = next(r for r in records if r["cut"] == "cubic111_gamma_typed_4_digits")
    require(typed["pyprocar_snap_note"] == [1, 1, 1], str(typed["box_text"]))
    random = [r for r in records if r["cut"].startswith("random")]
    require(
        min(r["angle_to_lattice_direction"] for r in random) > SNAP_ANGLE,
        f"a random cut lies within {SNAP_ANGLE} rad of a lattice direction",
    )
    return {
        "n_cuts": len(records),
        "seed": SEED,
        "worst_area_diff": max(abs(r["pyprocar_area"] - r["reference_area"]) for r in records),
        "worst_frequency_rel_diff": max(
            abs(r["pyprocar_frequency_gauss"] / r["reference_frequency_gauss"] - 1)
            for r in records
            if r["reference_frequency_gauss"]
        ),
        "n_cuts_with_orbits": sum(bool(r["reference_orbits"]) for r in records),
        "n_cuts_with_open_curves": sum(r["reference_open"] > 0 for r in records),
        "max_reference_reach": max(r["reference_reach"] for r in records),
        "snap_notes_on_exact_directions": [
            r["cut"]
            for r in records
            if r["pyprocar_snap_note"] and r["angle_to_lattice_direction"] <= LATTICE_NOISE
        ],
        "min_random_angle_to_lattice_direction": min(
            r["angle_to_lattice_direction"] for r in random
        ),
        "areas": {
            r["cut"]: [r["pyprocar_area"], round(r["reference_area"], 6), r["pyprocar_open"]]
            for r in records
        },
    }


def _zone_section(lattice: np.ndarray, plane: Plane, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Corners (M, 2) of the first zone's section by the plane, in the (u, v) frame at origin."""
    g = SHELL @ lattice
    halfspaces = np.column_stack([g @ u, g @ v, g @ plane.origin - 0.5 * (g * g).sum(axis=1)])
    halfspaces = halfspaces[np.abs(halfspaces[:, :2]).max(axis=1) > 1e-12]
    corners = HalfspaceIntersection(halfspaces, np.zeros(2)).intersections
    centre = corners.mean(axis=0)
    return corners[np.argsort(np.arctan2(*(corners - centre).T[::-1]))]


@step("png_reference_loops_on_zone")
def _():
    grid, fs = LOADED.grid, LOADED.fs
    assert grid is not None and fs is not None
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    sizes = {}
    for ax, name in zip(axes, ("cubic111_gamma", "cubic111_L"), strict=True):
        cut, normal, ref = REFERENCE_CUTS[name]
        u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        zone = _zone_section(grid.lattice, Plane(normal, cut.origin), u, v)
        closed = np.vstack([zone, zone[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.3", lw=1.2, label="first zone section")
        for i, loop in enumerate(ref.loops):
            rel = np.vstack([loop, loop[:1]]) - cut.origin
            ax.plot(
                rel @ u,
                rel @ v,
                color="tab:blue",
                lw=1.6,
                label="reference orbit" if i == 0 else None,
            )
        drawn = cast(pv.PolyData, fs.slice(normal=normal, origin=cut.origin))
        rel = np.asarray(drawn.points) - cut.origin
        ax.scatter(
            rel @ u, rel @ v, s=4, color="tab:orange", zorder=3, label="pyprocar drawn slice"
        )
        total = sum(ref.areas) * TWO_PI_SQUARED
        ax.set_title(f"{name}: reference {total:.4f} Ang^-2 (with 2 pi), {len(ref.areas)} orbit(s)")
        ax.set_aspect("equal")
        ax.set_xlabel("u (1/Ang, no 2 pi)")
        ax.set_ylabel("v (1/Ang, no 2 pi)")
        sizes[name] = int(drawn.n_points)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    size = png("reference_loops_on_zone", fig)
    require(
        size > 20_000 and min(sizes.values()) > 0, f"PNG {size} bytes, drawn slice points {sizes}"
    )
    return {"png_bytes": size, "drawn_slice_points": sizes}


finish()
