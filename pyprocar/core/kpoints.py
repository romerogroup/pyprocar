"""K-points and K-path handling for band structure calculations."""

from __future__ import annotations

__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.utils import math

logger = logging.getLogger(__name__)

# Type aliases for kpoints arrays
KPOINTS_DTYPE = npt.NDArray[np.float64]
RECIPROCAL_LATTICE_DTYPE = npt.NDArray[np.float64]


class KGRID_MODE(Enum):
    MONKHORST = "monkhorst"
    GAMMA = "gamma"


@dataclass
class KGridInfo:
    kgrid: tuple[int, int, int]
    kgrid_mode: KGRID_MODE
    kshift: tuple[float, float, float]


def generate_gamma_centered_kpoints(
    kgrid: tuple[int, int, int], kshift: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> npt.NDArray[np.float64]:
    n_kx = kgrid[0]
    n_ky = kgrid[1]
    n_kz = kgrid[2]

    kx_shift = kshift[0]
    ky_shift = kshift[1]
    kz_shift = kshift[2]

    kx_vals: npt.NDArray[np.float64] = (np.arange(0, n_kx) + kx_shift) / n_kx
    ky_vals: npt.NDArray[np.float64] = (np.arange(0, n_ky) + ky_shift) / n_ky
    kz_vals: npt.NDArray[np.float64] = (np.arange(0, n_kz) + kz_shift) / n_kz

    meshgrid: npt.NDArray[np.float64] = np.array(
        np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij")
    )
    move_axis: npt.NDArray[np.float64] = np.swapaxes(meshgrid, 0, -1)
    grid_points: npt.NDArray[np.float64] = move_axis.reshape(-1, 3)
    fbz_points: npt.NDArray[np.float64] = -np.fmod(grid_points + 6.5, 1) + 0.5
    sorted_kpoints = sort_kpoints(fbz_points, order="F")

    return sorted_kpoints


def monkhorst_pack_kpoints(
    kgrid: tuple[int, int, int], kshift: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> npt.NDArray[np.float64]:
    n_kx = kgrid[0]
    n_ky = kgrid[1]
    n_kz = kgrid[2]

    kx_shift = kshift[0]
    ky_shift = kshift[1]
    kz_shift = kshift[2]

    kx_vals: npt.NDArray[np.float64] = (np.arange(0, n_kx) + kx_shift + (1 - n_kx) / 2) / n_kx
    ky_vals: npt.NDArray[np.float64] = (np.arange(0, n_ky) + ky_shift + (1 - n_ky) / 2) / n_ky
    kz_vals: npt.NDArray[np.float64] = (np.arange(0, n_kz) + kz_shift + (1 - n_kz) / 2) / n_kz

    kx_grid, ky_grid, kz_grid = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij")

    result: npt.NDArray[np.float64] = np.stack(
        [kx_grid.flatten(), ky_grid.flatten(), kz_grid.flatten()], axis=-1
    )
    return result


def get_kpoints_from_kgrid(
    kgrid: tuple[int, int, int],
    kshift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: str = "monkhorst",
) -> npt.NDArray[np.float64]:
    if mode.lower()[0] == "m":
        return monkhorst_pack_kpoints(kgrid, kshift)
    elif mode.lower()[0] == "g":
        return generate_gamma_centered_kpoints(kgrid, kshift)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def reduced_to_cartesian(
    kpoints: npt.NDArray[np.float64], reciprocal_lattice: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64] | None:
    if reciprocal_lattice is not None:
        result: npt.NDArray[np.float64] = np.dot(kpoints, reciprocal_lattice)
        return result
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return None


def sort_kpoints(kpoints: npt.NDArray[np.float64], order: str = "C") -> npt.NDArray[np.float64]:
    sorted_indices: npt.NDArray[np.intp]
    if order == "C":
        sorted_indices = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
    elif order == "F":
        sorted_indices = np.lexsort((kpoints[:, 0], kpoints[:, 1], kpoints[:, 2]))
    else:
        raise ValueError(f"Invalid order: {order}")
    result: npt.NDArray[np.float64] = kpoints[sorted_indices]
    return result


def cartesian_to_reduced(
    cartesian: npt.NDArray[np.float64], reciprocal_lattice: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64] | None:
    """Converts cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    cartesian : np.ndarray
        The cartesian coordinates. shape = [N,3]
    reciprocal_lattice : np.ndarray
        The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None

    Returns
    -------
    np.ndarray
        The fractional coordinates. shape = [N,3]
    """
    if reciprocal_lattice is not None:
        kpoints: npt.NDArray[np.float64] = np.dot(cartesian, np.linalg.inv(reciprocal_lattice))
        return kpoints
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return None


def format_names(names: Sequence[str], as_latex: bool = False) -> list[str]:
    new_names: list[str] = []
    for x in names:
        name = x
        if name.lower() == "gamma":
            name = r"\Gamma"

        if "\\" in name and as_latex:
            name = "$" + name + "$"
        new_names.append(name)
    return new_names


SPECIAL_KPOINT_ALIASES = {
    "Γ": ["gamma", "Gamma", "G", "g", "Γ"],
    "X": ["x", "X"],
    "M": ["m", "M", "M-point"],
    "K": ["k", "K"],
    "L": ["l", "L"],
    # add more as needed
}


def normalize_kpoint_name(name: str) -> str:
    """Normalize special kpoint names to a canonical form."""
    name = name.strip()  # remove whitespace
    for canonical, aliases in SPECIAL_KPOINT_ALIASES.items():
        if name in aliases:
            return canonical
    return name  # fallback: return as-is if not found


class KPath:
    _kpoints: npt.NDArray[np.float64]
    _n_grids: list[int] | None
    _segment_names: list[tuple[str, str]] | None
    _tick_name_map: dict[int, str] | None
    _reciprocal_lattice: npt.NDArray[np.float64] | None
    _segment_indices: list[npt.NDArray[np.intp]]
    _continuous_start_indices: list[int]
    _discontinuity_start_indices: list[int]
    _special_kpoint_names: list[str]
    discontinuity_threshold: float
    zero_diff_threshold: float

    def __init__(
        self,
        kpoints: npt.NDArray[np.float64] | None = None,
        n_grids: list[int] | None = None,
        segment_names: list[tuple[str, str]] | None = None,
        special_kpoint_map: dict[str, npt.NDArray[np.float64]] | None = None,
        tick_name_map: dict[int, str] | None = None,
        reciprocal_lattice: npt.NDArray[np.float64] | None = None,
        discontinuity_threshold: float = 0.2,
        zero_diff_threshold: float = 1e-6,
        as_latex: bool = True,
    ) -> None:
        """
        The Kpath object to handle labels and ticks for band structure.

        Parameters
        ----------
        kpoints: np.ndarray
            The kpoints to be used for the kpath
        n_grids: List[int]
            The number of grids to be used for each segment
        segment_names: List[Tuple[str, str]]
            This is a list of tuples containing the names of the segments.
            The first element of the tuple is the name of the start point of the segment
            and the second element is the name of the end point of the segment.
        special_kpoint_map: Dict[str, np.ndarray]
            A dictionary containing the special kpoints.
            The key is the name of the special kpoint and the value is the kpoint.
        tick_name_map: Dict[int, str]
            A dictionary containing the names of ticks on the kpath.
            The key is the index of the tick and the value is the name of the tick.
        reciprocal_lattice: np.ndarray
            The reciprocal lattice of the crystal
        discontinuity_threshold: float
            The threshold for a discontinuity
        zero_diff_threshold: float
            The threshold for a zero difference
        """
        logger.info("Initializing KPath")
        logger.debug(f"n_grids: {n_grids}")
        logger.debug(f"discontinuity_threshold: {discontinuity_threshold}")
        logger.debug(f"zero_diff_threshold: {zero_diff_threshold}")
        logger.debug(f"tick_name_map: {tick_name_map}")
        logger.debug(f"reciprocal_lattice: \n {reciprocal_lattice}")

        if kpoints is None and n_grids is None:
            err_msg = "Either kpoints or n_grids must be provided"
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._n_grids = n_grids
        self.discontinuity_threshold = discontinuity_threshold
        self.zero_diff_threshold = zero_diff_threshold
        self._tick_name_map = tick_name_map
        self._reciprocal_lattice = reciprocal_lattice

        # Normalizing kpoint names to canonical form
        if segment_names is not None:
            segment_names = self._normalize_kpoint_names(segment_names)
        self._segment_names = segment_names

        # Generate kpoints if not provided
        if kpoints is not None:
            self._kpoints = kpoints
        else:
            if segment_names is None:
                raise ValueError("segment_names must be provided when kpoints is None")
            if special_kpoint_map is None:
                raise ValueError("special_kpoint_map must be provided when kpoints is None")
            if n_grids is None:
                raise ValueError("n_grids must be provided when kpoints is None")
            self._kpoints = self.generate_points(segment_names, special_kpoint_map, n_grids)
        logger.debug(f"Kpoints shape: {self._kpoints.shape}")

        # Get kpoint indices per kpath segment
        self._segment_indices, self._continuous_start_indices, self._discontinuity_start_indices = (
            self.get_segment_indices()
        )

        # Get unique special kpoint names
        self._special_kpoint_names = self.get_special_kpoint_names(
            segment_names=self._segment_names
        )

        # Format special kpoint names
        self.special_kpoint_names = format_names(self._special_kpoint_names, as_latex=as_latex)

        logger.info(f"\n{self}\n")
        logger.info("KPath initialized")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KPath):
            return False
        segment_names_equal = self.segment_names == other.segment_names
        special_kpoints_equal = bool(np.allclose(self.special_kpoints, other.special_kpoints))
        n_grids_equal = self.n_grids == other.n_grids
        tick_names_equal = self.tick_names == other.tick_names
        return segment_names_equal and special_kpoints_equal and n_grids_equal and tick_names_equal

    def __str__(self) -> str:
        ret = "K-Path\n"
        ret += "------\n"

        if self.segment_names is not None:
            for isegment, _ in enumerate(self.segment_indices):
                start_name, end_name = self.segment_names[isegment]
                start_kpoint = self.special_kpoint_map[start_name]
                end_kpoint = self.special_kpoint_map[end_name]

                ret += f"{isegment + 1:>2}. {start_name:<8}: ({start_kpoint[0]:>6.2f} {start_kpoint[1]:>6.2f} {start_kpoint[2]:>6.2f}) -> {end_name:<8}: ({end_kpoint[0]:>6.2f} {end_kpoint[1]:>6.2f} {end_kpoint[2]:>6.2f})\n"

        ret += "\n"
        ret += "Tick names:    " + "    ".join(f"{name:^8}" for name in self.tick_names) + "\n"
        ret += "Tick positions:" + "    ".join(f"{pos:^8}" for pos in self.tick_positions) + "\n"
        ret += "n_kpoints: " + str(self.n_kpoints) + "\n"
        ret += "n_segments: " + str(self.n_segments) + "\n"
        ret += "n_grids: " + str(self.n_grids) + "\n"
        ret += "discontinuity_indices: " + str(self.discontinuity_start_indices) + "\n"
        ret += "continuous_indices: " + str(self.continuous_start_indices) + "\n"

        return ret

    @property
    def n_kpoints(self) -> int:
        return len(self._kpoints)

    @property
    def n_grids(self) -> list[int] | None:
        return self._n_grids

    @property
    def n_segments(self) -> int:
        """The number of band segments.

        Returns
        -------
        int
            The number of band segments
        """
        return len(self.segment_indices)

    @property
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        return self._reciprocal_lattice

    @property
    def brillouin_zone(self) -> BrillouinZone:
        if self._reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice must be set to compute brillouin_zone")
        return BrillouinZone(self._reciprocal_lattice, transformation_matrix=[1, 1, 1])

    @property
    def kpoints(self) -> npt.NDArray[np.float64]:
        return self._kpoints

    @property
    def k_distances(self) -> npt.NDArray[np.float64]:
        distances = self.get_distances(as_segments=False)
        # When as_segments=False, get_distances always returns an array
        assert isinstance(distances, np.ndarray)
        return distances

    @property
    def segment_indices(self) -> list[npt.NDArray[np.intp]]:
        return self._segment_indices

    @property
    def knames(self) -> list[tuple[str, str]] | None:
        return self.segment_names

    @property
    def continuous_start_indices(self) -> list[int]:
        return self._continuous_start_indices

    @property
    def discontinuity_start_indices(self) -> list[int]:
        return self._discontinuity_start_indices

    @property
    def kpoints_cartesian(self) -> npt.NDArray[np.float64] | None:
        return reduced_to_cartesian(self.kpoints, self._reciprocal_lattice)

    @property
    def segment_names(self) -> list[tuple[str, str]] | None:
        return self._segment_names

    @segment_names.setter
    def segment_names(self, segment_names: list[tuple[str, str]]) -> None:
        if len(segment_names) != self.n_segments:
            raise ValueError(
                f"Number of segment names must match number of segments. Got {len(segment_names)} names for {self.n_segments} segments"
            )
        self._segment_names = segment_names
        # Note: _special_kpoints is a cached_property, this line won't work as expected
        # self._special_kpoints = self.get_special_kpoints()

    @property
    def special_kpoint_names(self) -> list[str]:
        return self._special_kpoint_names

    @special_kpoint_names.setter
    def special_kpoint_names(self, special_kpoint_names: list[str]) -> None:
        if len(special_kpoint_names) != len(self._special_kpoint_names):
            raise ValueError(
                f"Setting special kpoint names must match the existing number of special kpoint names.\n"
                f"Got {len(special_kpoint_names)} special kpoint names for {len(self._special_kpoint_names)} special kpoint names"
            )
        new_segment_names: list[tuple[str, str]] = []
        if self._segment_names is not None:
            for segment_name_tuple in self._segment_names:
                start_name, end_name = segment_name_tuple
                for j, special_kpoint in enumerate(self._special_kpoint_names):
                    if special_kpoint == start_name:
                        start_name = special_kpoint_names[j]
                    if special_kpoint == end_name:
                        end_name = special_kpoint_names[j]

                new_segment_names.append((start_name, end_name))

        self._segment_names = new_segment_names
        self._special_kpoint_names = special_kpoint_names

    @cached_property
    def special_kpoints(self) -> npt.NDArray[np.float64]:
        return self.get_special_kpoints(as_segments=True)

    @property
    def special_kpoint_map(self) -> dict[str, npt.NDArray[np.float64]]:
        special_kpoint_map: dict[str, npt.NDArray[np.float64]] = {}
        special_kpoints = self.get_special_kpoints(as_segments=False)
        for name, kpoint in zip(self.special_kpoint_names, special_kpoints):
            special_kpoint_map[name] = kpoint
        return special_kpoint_map

    def get_special_kpoint_names(
        self, segment_names: list[tuple[str, str]] | None = None
    ) -> list[str]:
        if segment_names is None:
            segment_names = self._segment_names
        special_kpoint_names: list[str] = []
        if segment_names is not None:
            for segment_name in segment_names:
                if segment_name[0] not in special_kpoint_names:
                    special_kpoint_names.append(segment_name[0])
                if segment_name[1] not in special_kpoint_names:
                    special_kpoint_names.append(segment_name[1])

        return special_kpoint_names

    def get_special_kpoints(
        self, as_segments: bool = False, cartesian: bool = False
    ) -> npt.NDArray[np.float64]:
        special_kpoints: list[Any] = []
        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        if kpoints is None:
            return np.array([])
        for segment_indices in self.segment_indices:
            start_kpoint = kpoints[segment_indices[0]]
            end_kpoint = kpoints[segment_indices[-1]]

            if as_segments:
                special_kpoints.append((start_kpoint, end_kpoint))
                continue

            # Check if start_kpoint is already in the list (using numpy array comparison)
            start_exists = any(np.allclose(start_kpoint, existing) for existing in special_kpoints)
            if not start_exists:
                special_kpoints.append(start_kpoint)

            # Check if end_kpoint is already in the list (using numpy array comparison)
            end_exists = any(np.allclose(end_kpoint, existing) for existing in special_kpoints)
            if not end_exists:
                special_kpoints.append(end_kpoint)

        result: npt.NDArray[np.float64] = np.array(special_kpoints)
        return result

    @property
    def tick_names_latex(self) -> list[str]:
        tick_names_latex: list[str] = []
        for tick_name in self.tick_names:
            name = tick_name
            if "\\" in name:
                name = f"${name}$"
            tick_names_latex.append(name)
        return tick_names_latex

    @property
    def tick_name_map(self) -> dict[int, str]:
        """The list of tick names.

        Returns
        -------
        dict[int, str]
            The mapping of tick indices to tick names
        """
        if self._tick_name_map is None:
            tick_name_map: dict[int, str] = {}
            if self._segment_names is None:
                self._tick_name_map = tick_name_map
                return self._tick_name_map
            for i, segment_indices in enumerate(self._segment_indices):
                start_index = int(segment_indices[0])
                end_index = int(segment_indices[-1])
                if i == 0:
                    tick_name_map[start_index] = self._segment_names[i][0]
                    continue
                if i == len(self._segment_indices) - 1:
                    tick_name_map[end_index] = self._segment_names[i][1]
                    continue

                if end_index in self.discontinuity_start_indices:
                    tick_name_map[end_index] = (
                        self._segment_names[i][0] + "|" + self._segment_names[i][1]
                    )

                    # Remove the previous segment end index. To avoid double tick
                    previous_segment_end_index = int(self._segment_indices[i - 1][-1])
                    tick_name_map.pop(previous_segment_end_index, None)
                elif end_index in self.continuous_start_indices:
                    tick_name_map[end_index] = self._segment_names[i][1]
                else:
                    raise ValueError(
                        f"Segment {i} is not a discontinuity or continuous segment. Likely a bug in get_segment_indices"
                    )

            self._tick_name_map = tick_name_map

        return self._tick_name_map

    @property
    def tick_names(self) -> list[str]:
        return list(self.tick_name_map.values())

    @property
    def tick_positions(self) -> list[int]:
        return list(self.tick_name_map.keys())

    def get_segments(
        self, isegments: Sequence[int] | None = None, cartesian: bool = False
    ) -> list[npt.NDArray[np.float64]]:
        if isegments is None:
            isegments = list(range(self.n_segments))

        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        if kpoints is None:
            return []

        segments: list[npt.NDArray[np.float64]] = []
        for segment_indices in self.segment_indices:
            segments.append(kpoints[segment_indices])

        return [segments[i] for i in isegments]

    def get_distances(
        self,
        isegments: Sequence[int] | None = None,
        as_segments: bool = True,
        cumlative_across_segments: bool = True,
        cartesian: bool = False,
    ) -> list[npt.NDArray[np.float64]] | npt.NDArray[np.float64]:
        segments = self.get_segments(isegments=isegments, cartesian=cartesian)

        k_segment_distances: list[npt.NDArray[np.float64]] = []
        previous_segment_max: float = 0.0
        for segment in segments:
            k_diffs: npt.NDArray[np.float64] = np.diff(segment, axis=0)
            k_diff_norms: npt.NDArray[np.float64] = np.linalg.norm(k_diffs, axis=1)
            k_distances: npt.NDArray[np.float64] = np.cumsum(k_diff_norms)

            k_distances = np.insert(k_distances, 0, 0)
            if cumlative_across_segments:
                k_distances = k_distances + previous_segment_max
                previous_segment_max = float(k_distances[-1])

            k_segment_distances.append(k_distances)

        if as_segments:
            return k_segment_distances
        else:
            return np.concatenate(k_segment_distances)

    def get_segment_indices(
        self,
    ) -> tuple[list[npt.NDArray[np.intp]], list[int], list[int]]:
        if len(self._kpoints) == 0:
            return [], [], []

        # Compute differences between consecutive kpoints
        k_diffs: npt.NDArray[np.float64] = np.diff(self._kpoints, axis=0)

        # Calculate the norm of differences
        k_diff_norms: npt.NDArray[np.float64] = np.linalg.norm(k_diffs, axis=1)

        # Find indices where difference is 0 (or very close to 0)
        continuous_end_indices: list[int] = [
            int(i) for i in np.where(k_diff_norms < self.zero_diff_threshold)[0]
        ]
        discontinuity_end_indices: list[int] = [
            int(i) for i in np.where(k_diff_norms > self.discontinuity_threshold)[0]
        ]

        segment_end_indices: list[int] = (
            continuous_end_indices + discontinuity_end_indices + [len(self._kpoints) - 1]
        )
        segment_end_indices.sort()

        indices: list[npt.NDArray[np.intp]] = []
        for i in range(len(segment_end_indices)):
            if i == 0:
                indices.append(np.arange(0, segment_end_indices[i] + 1))
            else:
                indices.append(
                    np.arange(segment_end_indices[i - 1] + 1, segment_end_indices[i] + 1)
                )

        return indices, continuous_end_indices, discontinuity_end_indices

    def get_continuous_segments(self) -> list[npt.NDArray[np.intp]]:
        continuous_segments: list[npt.NDArray[np.intp]] = []
        for isegment, segment_indices in enumerate(self.segment_indices):
            if isegment == 0:
                continuous_segments.append(segment_indices)
                continue

            previous_segment_end_index = int(self.segment_indices[isegment - 1][-1])

            if previous_segment_end_index in self.discontinuity_start_indices:
                continuous_segments.append(segment_indices)
                continue

            elif previous_segment_end_index in self.continuous_start_indices:
                continuous_segments[-1] = np.concatenate((continuous_segments[-1], segment_indices))
                continue
            else:
                raise ValueError(
                    f"Segment {isegment} is not a discontinuity or continuous segment. Likely a bug in get_segment_indices"
                )

        return continuous_segments

    def generate_points(
        self,
        segment_names: list[tuple[str, str]],
        special_kpoints_map: dict[str, npt.NDArray[np.float64]],
        n_grids: list[int],
    ) -> npt.NDArray[np.float64]:
        """Generate the kpath points from segment definitions.

        Parameters
        ----------
        segment_names : list[tuple[str, str]]
            List of tuples containing the names of the segments.
        special_kpoints_map : dict[str, npt.NDArray[np.float64]]
            A dictionary containing the special kpoints.
        n_grids : list[int]
            The number of grid points for each segment.

        Returns
        -------
        npt.NDArray[np.float64]
            The generated kpoints along the path.
        """
        logger.info("Generating kpoints from special kpoints and ngrids")

        kpoints_on_path: npt.NDArray[np.float64] | None = None
        for isegment, segment_name in enumerate(segment_names):
            kstart_label, kend_label = segment_name
            kstart = special_kpoints_map[kstart_label]
            kend = special_kpoints_map[kend_label]
            kpoints: npt.NDArray[np.float64] = np.linspace(kstart, kend, n_grids[isegment])

            if kpoints_on_path is None:
                kpoints_on_path = kpoints
            else:
                kpoints_on_path = np.concatenate((kpoints_on_path, kpoints))

        if kpoints_on_path is None:
            return np.array([], dtype=np.float64)
        return kpoints_on_path

    def _normalize_kpoint_names(
        self, segment_names: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        new_segment_names: list[tuple[str, str]] = []
        for segment_name in segment_names:
            kstart_label, kend_label = segment_name
            normalized_kstart_label = normalize_kpoint_name(kstart_label)
            normalized_kend_label = normalize_kpoint_name(kend_label)
            new_segment_names.append((normalized_kstart_label, normalized_kend_label))
        return new_segment_names

    def get_optimized_kpoints_transformed(
        self, transformation_matrix: npt.NDArray[np.float64], same_grid_size: bool = False
    ) -> KPath:
        """Get the optimized kpoints after a transformation.

        Parameters
        ----------
        transformation_matrix : npt.NDArray[np.float64]
            The transformation matrix.
        same_grid_size : bool
            Boolean to determine if the grid should retain the same size

        Returns
        -------
        KPath
            The transformed KPath
        """
        if self.n_grids is None:
            raise ValueError("n_grids must be set to use get_optimized_kpoints_transformed")

        new_special_kpoints: npt.NDArray[np.float64] = np.dot(
            self.special_kpoints, transformation_matrix
        )
        new_ngrids: list[int] = self.n_grids.copy()
        for isegment in range(self.n_segments):
            # Extract segment endpoints
            segment_start_new: npt.NDArray[np.float64] = np.asarray(
                new_special_kpoints[isegment][0], dtype=np.float64
            )
            segment_end_new: npt.NDArray[np.float64] = np.asarray(
                new_special_kpoints[isegment][1], dtype=np.float64
            )
            segment_start_old: npt.NDArray[np.float64] = np.asarray(
                self.special_kpoints[isegment][0], dtype=np.float64
            )
            segment_end_old: npt.NDArray[np.float64] = np.asarray(
                self.special_kpoints[isegment][1], dtype=np.float64
            )
            kstart: npt.NDArray[np.float64] = segment_start_new
            kend: npt.NDArray[np.float64] = segment_end_new
            kpoints_old: npt.NDArray[np.float64] = np.linspace(
                segment_start_old,
                segment_end_old,
                self.n_grids[isegment],
            )

            dk_vector_old: npt.NDArray[np.float64] = kpoints_old[-1] - kpoints_old[-2]
            dk_old: float = float(np.linalg.norm(dk_vector_old))

            # this part is to find the direction
            distance: npt.NDArray[np.float64] = kend - kstart

            # this part is to find the high symmetry points on the path
            expand: npt.NDArray[np.float64] = (np.linspace(kstart, kend, 1000) * 2).round(0) / 2

            unique_indexes: npt.NDArray[np.intp] = np.sort(
                np.unique(expand, return_index=True, axis=0)[1]
            )
            symm_kpoints_path: npt.NDArray[np.float64] = expand[unique_indexes]

            # this part is to only select points that are after kstart and not before
            angles: npt.NDArray[np.float64] = np.array(
                [math.get_angle(x, distance, radians=False) for x in (symm_kpoints_path - kstart)]
            ).round()
            symm_kpoints_path = symm_kpoints_path[angles == 0]
            if len(symm_kpoints_path) < 2:
                continue
            suggested_kstart: npt.NDArray[np.float64] = symm_kpoints_path[0]
            suggested_kend: npt.NDArray[np.float64] = symm_kpoints_path[1]

            if np.linalg.norm(distance) > np.linalg.norm(suggested_kend - suggested_kstart):
                new_special_kpoints[isegment][0] = suggested_kstart
                new_special_kpoints[isegment][1] = suggested_kend

            # this part is to get the number of grid points in the to have the
            # same spacing as before the transformation
            if same_grid_size:
                new_ngrids[isegment] = int(
                    (
                        np.linalg.norm(
                            new_special_kpoints[isegment][0] - new_special_kpoints[isegment][1]
                        )
                        / dk_old
                    ).round(4)
                    + 1
                )
        return KPath(kpoints=new_special_kpoints, n_grids=new_ngrids)

    def get_kpoints_transformed(
        self,
        transformation_matrix: npt.NDArray[np.float64],
    ) -> KPath:
        """Get the transformed kpoints.

        Parameters
        ----------
        transformation_matrix : npt.NDArray[np.float64]
            The transformation matrix

        Returns
        -------
        KPath
            The transformed KPath
        """
        new_special_kpoints: npt.NDArray[np.float64] = np.dot(
            self.special_kpoints, transformation_matrix
        )
        return KPath(kpoints=new_special_kpoints, reciprocal_lattice=transformation_matrix)

    def write_to_file(self, filename: str = "KPOINTS", fmt: str = "vasp") -> None:
        """Write the kpath to a file. Only supports vasp at the moment.

        Parameters
        ----------
        filename : str, optional
            The output filename, by default "KPOINTS"
        fmt : str, optional
            The output format, by default "vasp"
        """
        if self.n_grids is None:
            raise ValueError("n_grids must be set to write to file")
        if self._segment_names is None:
            raise ValueError("segment_names must be set to write to file")

        with open(filename, "w") as wf:
            if fmt == "vasp":
                wf.write("! Generated by pyprocar\n")
                if len(np.unique(self.n_grids)) == 1:
                    wf.write(str(self.n_grids[0]) + "\n")
                else:
                    wf.write("   ".join([str(x) for x in self.n_grids]) + "\n")
                wf.write("Line-mode\n")
                wf.write("reciprocal\n")
                for isegment in range(self.n_segments):
                    segment_name = self._segment_names[isegment]
                    wf.write(
                        " ".join([f"  {x:8.4f}" for x in self.special_kpoints[isegment][0]])
                        + "   ! "
                        + segment_name[0].replace("$", "")
                        + "\n"
                    )
                    wf.write(
                        " ".join([f"  {x:8.4f}" for x in self.special_kpoints[isegment][1]])
                        + "   ! "
                        + segment_name[1].replace("$", "")
                        + "\n"
                    )
                    wf.write("\n")

    def plot(
        self,
        add_point_labels_args: dict[str, Any] | None = None,
        bz_add_mesh_args: dict[str, Any] | None = None,
        as_cartesian: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot the k-path in the Brillouin zone."""
        add_point_labels_args = add_point_labels_args or {}
        bz_add_mesh_args = bz_add_mesh_args or {}

        p = pv.Plotter()

        if as_cartesian:
            kpath = pv.PolyData(self.kpoints_cartesian)
        else:
            kpath = pv.PolyData(self.kpoints)

        p.add_mesh(kpath, **kwargs)

        special_kpoints = self.get_special_kpoints(as_segments=False, cartesian=as_cartesian)
        special_kpoint_names = self.get_special_kpoint_names()
        p.add_point_labels(special_kpoints, special_kpoint_names, **add_point_labels_args)

        bz_add_mesh_args["style"] = bz_add_mesh_args.get("style", "wireframe")
        bz_add_mesh_args["line_width"] = bz_add_mesh_args.get("line_width", 2.0)
        bz_add_mesh_args["color"] = bz_add_mesh_args.get("color", "black")
        bz_add_mesh_args["opacity"] = bz_add_mesh_args.get("opacity", 1.0)

        p.add_mesh(
            self.brillouin_zone,
            **bz_add_mesh_args,
        )
        p.show()
