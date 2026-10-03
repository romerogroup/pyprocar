__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"


import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Literal

import numpy as np
import pyvista as pv

from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.utils import math, np_utils

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

KPOINTS_DTYPE = np.ndarray[tuple[int, Literal[3]], np.dtype[np_utils.FLOAT_DTYPE]]
RECIPROCAL_LATTICE_DTYPE = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np_utils.FLOAT_DTYPE]]


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
):
    n_kx = kgrid[0]
    n_ky = kgrid[1]
    n_kz = kgrid[2]

    kx_shift = kshift[0]
    ky_shift = kshift[1]
    kz_shift = kshift[2]

    kx_vals = (np.arange(0, n_kx) + kx_shift) / n_kx
    ky_vals = (np.arange(0, n_ky) + ky_shift) / n_ky
    kz_vals = (np.arange(0, n_kz) + kz_shift) / n_kz

    meshgrid = np.array(np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij"))
    move_axis = np.swapaxes(meshgrid, 0, -1)
    grid_points = move_axis.reshape(-1, 3)
    fbz_points = -np.fmod(grid_points + 6.5, 1) + 0.5
    sorted_kpoints = sort_kpoints(fbz_points, order="F")

    return sorted_kpoints


def monkhorst_pack_kpoints(
    kgrid: tuple[int, int, int], kshift: tuple[float, float, float] = (0.0, 0.0, 0.0)
):
    n_kx = kgrid[0]
    n_ky = kgrid[1]
    n_kz = kgrid[2]

    kx_shift = kshift[0]
    ky_shift = kshift[1]
    kz_shift = kshift[2]

    kpoints = np.zeros((np.prod(kgrid), 3))
    kx_vals = (np.arange(0, n_kx) + kx_shift + (1 - n_kx) / 2) / n_kx
    ky_vals = (np.arange(0, n_ky) + ky_shift + (1 - n_ky) / 2) / n_ky
    kz_vals = (np.arange(0, n_kz) + kz_shift + (1 - n_kz) / 2) / n_kz

    kx_vals, ky_vals, kz_vals = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij")

    return np.stack([kx_vals.flatten(), ky_vals.flatten(), kz_vals.flatten()], axis=-1)


def get_kpoints_from_kgrid(
    kgrid: tuple[int, int, int],
    kshift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: str | KGRID_MODE = "monkhorst",
):
    if isinstance(mode, KGRID_MODE):
        mode = mode.value
    if mode.lower()[0] == "m":
        return monkhorst_pack_kpoints(kgrid, kshift)
    elif mode.lower()[0] == "g":
        return generate_gamma_centered_kpoints(kgrid, kshift)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def reduced_to_cartesian(
    kpoints: KPOINTS_DTYPE, reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE
) -> KPOINTS_DTYPE:
    if reciprocal_lattice is not None:
        return np.dot(kpoints, reciprocal_lattice)
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return None


def sort_kpoints(kpoints: KPOINTS_DTYPE, order: str = "C"):
    if order == "C":
        sorted_indices = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
    elif order == "F":
        sorted_indices = np.lexsort((kpoints[:, 0], kpoints[:, 1], kpoints[:, 2]))
    return kpoints[sorted_indices]


def cartesian_to_reduced(
    cartesian: KPOINTS_DTYPE, reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE
) -> KPOINTS_DTYPE:
    """Converts cartesian coordinates to fractional coordinates

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
        kpoints = np.dot(cartesian, np.linalg.inv(reciprocal_lattice))
        return kpoints
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return None


def format_names(names: list[str], as_latex: bool = False):
    new_names = []
    for x in names:
        if x.lower() == "gamma":
            x = r"\Gamma"

        if "\\" in x and as_latex:
            x = "$" + x + "$"
        new_names.append(x)
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
    def __init__(
        self,
        kpoints: np.ndarray = None,
        n_grids: list[int] = None,
        segment_names: list[tuple[str, str]] = None,
        special_kpoint_map: dict[str, np.ndarray] = None,
        tick_name_map: dict[int, str] = None,
        reciprocal_lattice=None,
        discontinuity_threshold=0.2,
        zero_diff_threshold=1e-6,
        as_latex=True,
        segment_end_indices: list[int] | None = None,
    ):
        """
        The Kpath object to handle labels and ticks for band structure

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
            With kpoints given, it measures the first step after a jump from the
            next segment's start point, for codes that leave that point out.
        tick_name_map: Dict[int, str]
            A dictionary containing the names of ticks on the kpath.
            The key is the index of the tick and the value is the name of the tick.
        reciprocal_lattice: np.ndarray
            The reciprocal lattice of the crystal
        discontinuity_threshold: float
            The threshold for a discontinuity
        zero_diff_threshold: float
            The threshold for a zero difference
        segment_end_indices: List[int], optional
            The index of the last k-point of each segment. Use it when the
            k-points do not repeat segment boundaries, so the segments cannot
            be found from the k-points alone. By default the segments are found
            from repeated k-points and jumps.
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
        self._segment_end_indices: list[int] | None = segment_end_indices

        # Normalizing kpoint names to canonical form
        raw_segment_names = segment_names
        segment_names = self._normalize_kpoint_names(segment_names)
        self._segment_names = segment_names

        # Generate kpoints if not provided
        self._kpoints = kpoints
        named_starts: list[np.ndarray | None] = []
        if self._kpoints is None:
            self._kpoints = self.generate_points(segment_names, special_kpoint_map, n_grids)
            self._segment_end_indices = (np.cumsum(n_grids) - 1).tolist()
        elif special_kpoint_map:
            named_starts = [special_kpoint_map.get(start) for start, _ in raw_segment_names]
        logger.debug(f"Kpoints shape: {self._kpoints.shape}")

        # Get kpoint indices per kpath segment
        self._segment_indices, self._continuous_start_indices, self._discontinuity_start_indices = (
            self.get_segment_indices()
        )
        if len(segment_names) != self.n_segments:
            user_logger.warning(
                "KPath got %d segment names for %d segments in the k-points; ticks use %d",
                len(segment_names),
                self.n_segments,
                min(len(segment_names), self.n_segments),
            )
            named_starts = []
        self._jump_start_kpoints: dict[int, np.ndarray] = self._get_jump_start_kpoints(named_starts)

        # Get unique special kpoint names
        self._special_kpoint_names = self.get_special_kpoint_names(
            segment_names=self._segment_names
        )

        # Format special kpoint names
        self.special_kpoint_names = format_names(self._special_kpoint_names, as_latex=as_latex)

        logger.info("\n%s\n", self)
        logger.info("KPath initialized")

    def __eq__(self, other):
        segment_names_equal = self.segment_names == other.segment_names
        special_kpoints_equal = np.allclose(self.special_kpoints, other.special_kpoints)
        n_grids_equal = self.n_grids == other.n_grids
        tick_names_equal = self.tick_names == other.tick_names
        return segment_names_equal and special_kpoints_equal and n_grids_equal and tick_names_equal

    def __str__(self):
        ret = "K-Path\n"
        ret += "------\n"

        for isegment, ((start_name, end_name), (start_kpoint, end_kpoint)) in enumerate(
            zip(self.segment_names, self.special_kpoints, strict=False)
        ):
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
    def n_kpoints(self):
        return len(self._kpoints)

    @property
    def n_grids(self):
        return self._n_grids

    @property
    def n_segments(self):
        """The number of band segments

        Returns
        -------
        int
            The number of band segments
        """
        return len(self.segment_indices)

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice

    @property
    def brillouin_zone(self):
        return BrillouinZone(self.reciprocal_lattice, transformation_matrix=[1, 1, 1])

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def k_distances(self):
        return self.get_distances(as_segments=False)

    @property
    def segment_indices(self):
        return self._segment_indices

    @property
    def knames(self):
        return self.segment_names

    @property
    def continuous_start_indices(self):
        return self._continuous_start_indices

    @property
    def discontinuity_start_indices(self):
        return self._discontinuity_start_indices

    @property
    def kpoints_cartesian(self):
        if self._reciprocal_lattice is None:
            raise ValueError("KPath needs a reciprocal_lattice for Cartesian k-points")
        return reduced_to_cartesian(self.kpoints, self._reciprocal_lattice)

    @property
    def segment_names(self):
        return self._segment_names

    @segment_names.setter
    def segment_names(self, segment_names):
        if len(segment_names) != self.n_segments:
            raise ValueError(
                f"Number of segment names must match number of segments. Got {len(segment_names)} names for {self.n_segments} segments"
            )
        self._segment_names = self._normalize_kpoint_names(segment_names)
        self._special_kpoint_names = self.get_special_kpoint_names(self._segment_names)
        self._tick_name_map = None

    @property
    def special_kpoint_names(self):
        return self._special_kpoint_names

    @special_kpoint_names.setter
    def special_kpoint_names(self, special_kpoint_names):
        if len(special_kpoint_names) != len(self._special_kpoint_names):
            raise ValueError(
                f"Setting special kpoint names must match the existing number of special kpoint names.\n"
                f"Got {len(special_kpoint_names)} special kpoint names for {len(self._special_kpoint_names)} special kpoint names"
            )
        new_segment_names = []
        for i, segment_name_tuple in enumerate(self._segment_names):
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
    def special_kpoints(self):
        return self.get_special_kpoints(as_segments=True)

    @property
    def special_kpoint_map(self):
        special_kpoint_map = {}
        special_kpoints = self.get_special_kpoints(as_segments=False)
        for name, kpoint in zip(self.special_kpoint_names, special_kpoints):
            special_kpoint_map[name] = kpoint
        return special_kpoint_map

    def get_special_kpoint_names(self, segment_names: list[tuple[str, str]] = None):
        if segment_names is None:
            segment_names = self._segment_names
        special_kpoint_names = []
        for i, segment_name in enumerate(segment_names):
            if segment_name[0] not in special_kpoint_names:
                special_kpoint_names.append(segment_name[0])
            if segment_name[1] not in special_kpoint_names:
                special_kpoint_names.append(segment_name[1])

        return special_kpoint_names

    def get_special_kpoints(self, as_segments: bool = False, cartesian: bool = False):
        special_kpoints = []
        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        start_index = 0
        for segment_indices in self.segment_indices:
            start_kpoint = kpoints[start_index]
            end_kpoint = kpoints[segment_indices[-1]]
            start_index = segment_indices[-1]
            if start_index in self.discontinuity_start_indices:
                start_index += 1

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

        return np.array(special_kpoints)

    @property
    def tick_names_latex(self):
        tick_names_latex = []
        for tick_index, tick_name in enumerate(self.tick_names):
            if "\\" in tick_name:
                tick_name = f"${tick_name}$"
            tick_names_latex.append(tick_name)
        return tick_names_latex

    @property
    def tick_name_map(self):
        """The list of tick names

        Returns
        -------
        List
            The list of tick names
        """
        if self._tick_name_map is None:
            names = self._segment_names
            tick_name_map = {self._segment_indices[0][0]: names[0][0]}
            for i, segment_indices in enumerate(self._segment_indices[: len(names)]):
                end_index = segment_indices[-1]
                name = names[i][1]
                if end_index in self.discontinuity_start_indices and i + 1 < len(names):
                    name += "|" + names[i + 1][0]
                tick_name_map[end_index] = name

            self._tick_name_map = tick_name_map

        return self._tick_name_map

    @property
    def tick_names(self):
        return list(self.tick_name_map.values())

    @property
    def tick_positions(self):
        return list(self.tick_name_map.keys())

    def get_segments(self, isegments: list[int] = None, cartesian: bool = False):
        if isegments is None:
            isegments = list(range(self.n_segments))

        kpoints = self.kpoints_cartesian if cartesian else self.kpoints

        segments = []
        for segment_indices in self.segment_indices:
            segments.append(kpoints[segment_indices])

        return [segments[i] for i in isegments]

    def _get_jump_start_kpoints(
        self, named_starts: list[np.ndarray | None]
    ) -> dict[int, np.ndarray]:
        """Named start point of each segment that follows a jump, by jump index.

        A code may leave out the first point after a jump (Abinit), so the
        step across the jump is measured from the named start point. A name
        is trusted only when its point is the segment's first k-point or one
        step before it; otherwise the jump counts as zero.
        """
        jump_starts: dict[int, np.ndarray] = {}
        if not named_starts:
            return jump_starts
        for isegment in range(1, self.n_segments):
            jump_index = self.segment_indices[isegment - 1][-1]
            if jump_index not in self.discontinuity_start_indices:
                continue
            start_name = self._segment_names[isegment][0]
            named = named_starts[isegment]
            if named is None:
                user_logger.warning(
                    "KPath has no point for %s; the jump before it counts as zero", start_name
                )
                continue
            named = np.asarray(named, dtype=float)
            segment = self._kpoints[self.segment_indices[isegment]]
            candidates = [segment[0]]
            if len(segment) > 1:
                candidates.append(2 * segment[0] - segment[1])
            if not any(np.allclose(named, c, atol=1e-4) for c in candidates):
                user_logger.warning(
                    "KPath point for %s does not match its segment; jump counts as zero",
                    start_name,
                )
                continue
            jump_starts[jump_index] = named
        return jump_starts

    def get_distances(
        self,
        isegments: list[int] | None = None,
        as_segments: bool = True,
        cumlative_across_segments: bool = True,
        cartesian: bool = True,
    ):
        if isegments is None:
            isegments = list(range(self.n_segments))

        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        steps = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
        steps[self.discontinuity_start_indices] = 0.0
        for jump_index, start in self._jump_start_kpoints.items():
            if cartesian:
                start = start @ self._reciprocal_lattice
            steps[jump_index] = np.linalg.norm(kpoints[jump_index + 1] - start)
        path_distances = np.insert(np.cumsum(steps), 0, 0.0)

        k_segment_distances = []
        for isegment in isegments:
            k_distances = path_distances[self.segment_indices[isegment]]
            if not cumlative_across_segments:
                k_distances = k_distances - k_distances[0]
            k_segment_distances.append(k_distances)

        if as_segments:
            return k_segment_distances
        else:
            return np.concatenate(k_segment_distances)

    def get_segment_indices(self):
        if self._kpoints is None or len(self._kpoints) == 0:
            return np.array([])

        # Compute differences between consecutive kpoints
        k_diffs = np.diff(self._kpoints, axis=0)

        # Calculate the norm of differences
        k_diff_norms = np.linalg.norm(k_diffs, axis=1)

        if self._segment_end_indices is None:
            is_boundary = (k_diff_norms < self.zero_diff_threshold) | (
                k_diff_norms > self.discontinuity_threshold
            )
            boundaries = list(np.where(is_boundary)[0])
        else:
            boundaries = self._segment_end_indices[:-1]
        is_jump = k_diff_norms > self.discontinuity_threshold
        continuous_end_indices = [i for i in boundaries if not is_jump[i]]
        discontinuity_end_indices = [i for i in boundaries if is_jump[i]]

        segment_end_indices = (
            continuous_end_indices + discontinuity_end_indices + [len(self._kpoints) - 1]
        )
        segment_end_indices.sort()

        indices = []
        for i, segment_end_index in enumerate(segment_end_indices):
            if i == 0:
                indices.append(np.arange(0, segment_end_indices[i] + 1))
            else:
                indices.append(
                    np.arange(segment_end_indices[i - 1] + 1, segment_end_indices[i] + 1)
                )

        return indices, continuous_end_indices, discontinuity_end_indices

    def get_continuous_segments(self):
        continuous_segments = []
        for isegment, segment_indices in enumerate(self.segment_indices):
            if isegment == 0:
                continuous_segments.append(segment_indices)
                continue

            previous_segment_end_index = self.segment_indices[isegment - 1][-1]

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
        segment_names: list[tuple[str, str]] = None,
        special_kpoints_map: dict[str, np.ndarray] = None,
        n_grids: list[int] = None,
    ):
        """
        Generate the kpath points
        """

        kpoints_on_path = []
        if self.kpoints is not None:
            for isegment in range(self.n_segments):
                kstart, kend = self.special_kpoints[isegment]
                kpoints = np.linspace(kstart, kend, self.n_grids[isegment])

                if len(kpoints_on_path) == 0:
                    kpoints_on_path = kpoints
                else:
                    kpoints_on_path = np.concatenate((kpoints_on_path, kpoints))

        else:
            logger.info("No kpoints provided. Generating kpoints from special kpoints and ngrids")

            if n_grids is None:
                raise ValueError("n_grids must be provided")
            if segment_names is None:
                raise ValueError("segment_names must be provided")
            if special_kpoints_map is None:
                raise ValueError("special_kpoints_map must be provided")

            for isegment, segment_name in enumerate(segment_names):
                kstart_label, kend_label = segment_name
                kstart = special_kpoints_map[kstart_label]
                kend = special_kpoints_map[kend_label]
                kpoints = np.linspace(kstart, kend, n_grids[isegment])

                if len(kpoints_on_path) == 0:
                    kpoints_on_path = kpoints
                else:
                    kpoints_on_path = np.concatenate((kpoints_on_path, kpoints))
        return kpoints_on_path

    def _normalize_kpoint_names(self, segment_names: list[tuple[str, str]]):
        new_segment_names = []
        for segment_name in segment_names:
            kstart_label, kend_label = segment_name
            normalized_kstart_label = normalize_kpoint_name(kstart_label)
            normalized_kend_label = normalize_kpoint_name(kend_label)
            new_segment_names.append((normalized_kstart_label, normalized_kend_label))
        return new_segment_names

    def get_optimized_kpoints_transformed(self, transformation_matrix, same_grid_size=False):
        """
        A method to get the optimized kpoints after a transformation

        Parameters
        ----------
        transformation_matrix : np.ndarray
            The transformmation matrix.
        same_grid_size : bool
            Boolean to determine if the grid should retain the same size

        Returns
        -------
        pyprocar.core.KPath
            The transformed KPath
        """

        new_special_kpoints = np.dot(self.special_kpoints, transformation_matrix)
        new_ngrids = self.n_grids.copy()
        for isegment in range(self.n_segments):
            kstart = new_special_kpoints[isegment][0]
            kend = new_special_kpoints[isegment][1]
            kpoints_old = np.linspace(
                self.special_kpoints[isegment][0],
                self.special_kpoints[isegment][1],
                self.n_grids[isegment],
            )

            dk_vector_old = kpoints_old[-1] - kpoints_old[-2]
            dk_old = np.linalg.norm(dk_vector_old)

            # this part is to find the direction
            distance = kend - kstart

            # this part is to find the high symmetry points on the path
            expand = (np.linspace(kstart, kend, 1000) * 2).round(0) / 2

            unique_indexes = np.sort(np.unique(expand, return_index=True, axis=0)[1])
            symm_kpoints_path = expand[unique_indexes]

            # this part is to only select poits that are after kstart and not before

            angles = np.array(
                [math.get_angle(x, distance, radians=False) for x in (symm_kpoints_path - kstart)]
            ).round()
            symm_kpoints_path = symm_kpoints_path[angles == 0]
            if len(symm_kpoints_path) < 2:
                continue
            suggested_kstart = symm_kpoints_path[0]
            suggested_kend = symm_kpoints_path[1]

            if np.linalg.norm(distance) > np.linalg.norm(suggested_kend - suggested_kstart):
                new_special_kpoints[isegment][0] = suggested_kstart
                new_special_kpoints[isegment][1] = suggested_kend

            # this part is to get the number of gird points in the to have the
            # same spacing is before the transformation
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
        return KPath(special_kpoints=new_special_kpoints, n_grids=new_ngrids)

    def get_kpoints_transformed(
        self,
        transformation_matrix,
    ):
        """A method to get the transformed kpoints

        Parameters
        ----------
        transformation_matrix : np.ndarray
            The transformation matrix

        Returns
        -------
        pyprocar.core.KPath
            The transformed KPath
        """
        new_special_kpoints = np.dot(self.special_kpoints, transformation_matrix)
        return KPath(kpoints=new_special_kpoints, reciprocal_lattice=transformation_matrix)

    def write_to_file(self, filename="KPOINTS", fmt="vasp"):
        """Write the kpath to a file. Only supports vasp at the moment

        Parameters
        ----------
        filename : str, optional
            _description_, by default "KPOINTS"
        fmt : str, optional
            _description_, by default "vasp"
        """
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
                    wf.write(
                        " ".join([f"  {x:8.4f}" for x in self.special_kpoints[isegment][0]])
                        + "   ! "
                        + self.special_kpoint_names[isegment][0].replace("$", "")
                        + "\n"
                    )
                    wf.write(
                        " ".join([f"  {x:8.4f}" for x in self.special_kpoints[isegment][1]])
                        + "   ! "
                        + self.special_kpoint_names[isegment][1].replace("$", "")
                        + "\n"
                    )
                    wf.write("\n")

        return None

    def plot(
        self,
        add_point_labels_args: dict = None,
        bz_add_mesh_args: dict = None,
        as_cartesian: bool = False,
        **kwargs,
    ):
        """
        Plots the band structure.

        """
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
