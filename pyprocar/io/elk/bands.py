"""BANDS.OUT and BANDLINES.OUT parser for Elk calculations."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

HARTREE_TO_EV = 27.211386245988


class ElkBands:
    """Parser for Elk BANDS.OUT and BANDLINES.OUT files.

    BANDS.OUT contains eigenvalues for each k-point and band.
    BANDLINES.OUT contains k-point positions along the band path.

    Parameters
    ----------
    bands_filepath : Path | None
        Path to BANDS.OUT file
    bandlines_filepath : Path | None
        Path to BANDLINES.OUT file
    bands_str : str
        Content of BANDS.OUT file (alternative to filepath)
    bandlines_str : str
        Content of BANDLINES.OUT file (alternative to filepath)
    nkpoints : int
        Number of k-points (from elk.in plot1d block)
    nspin : int
        Number of spin channels
    high_symmetry_points : np.ndarray
        High-symmetry point coordinates for k-point interpolation
    """

    def __init__(
        self,
        bands_filepath: Path | None = None,
        bandlines_filepath: Path | None = None,
        bands_str: str = "",
        bandlines_str: str = "",
        nkpoints: int = 0,
        nspin: int = 1,
        high_symmetry_points: npt.NDArray[np.float64] | None = None,
    ):
        self._bands_filepath: Path | None = bands_filepath
        self._bandlines_filepath: Path | None = bandlines_filepath
        self._bands_str: str = bands_str
        self._bandlines_str: str = bandlines_str
        self._nkpoints: int = nkpoints
        self._nspin: int = nspin
        self._high_symmetry_points: npt.NDArray[np.float64] = (
            high_symmetry_points if high_symmetry_points is not None else np.array([])
        )

    @classmethod
    def from_str(
        cls,
        bands_content: str,
        bandlines_content: str,
        nkpoints: int,
        nspin: int = 1,
        high_symmetry_points: npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Create parser from file content strings."""
        return cls(
            bands_str=bands_content,
            bandlines_str=bandlines_content,
            nkpoints=nkpoints,
            nspin=nspin,
            high_symmetry_points=high_symmetry_points,
        )

    @cached_property
    def bands_str(self) -> str:
        """Lazily load BANDS.OUT content."""
        if self._bands_str == "" and self._bands_filepath is not None:
            return Path(self._bands_filepath).read_text()
        elif self._bands_str == "" and self._bands_filepath is None:
            raise ValueError("No BANDS.OUT filepath or content provided")
        return self._bands_str

    @cached_property
    def bandlines_str(self) -> str:
        """Lazily load BANDLINES.OUT content."""
        if self._bandlines_str == "" and self._bandlines_filepath is not None:
            return Path(self._bandlines_filepath).read_text()
        elif self._bandlines_str == "" and self._bandlines_filepath is None:
            raise ValueError("No BANDLINES.OUT filepath or content provided")
        return self._bandlines_str

    @cached_property
    def nkpoints(self) -> int:
        """Number of k-points."""
        return self._nkpoints

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels."""
        return self._nspin

    @cached_property
    def _raw_nbands(self) -> int:
        """Raw number of bands (before spin separation)."""
        lines = self.bands_str.splitlines()
        return len(lines) // (self.nkpoints + 1)

    @cached_property
    def nbands(self) -> int:
        """Number of bands per spin channel."""
        if self.nspin == 1:
            return self._raw_nbands
        return self._raw_nbands // 2

    @cached_property
    def _tick_positions(self) -> list[str]:
        """K-point positions at high-symmetry points (as strings for exact comparison)."""
        lines = self.bandlines_str.splitlines()
        ticks: list[str] = []
        for i in range(0, len(lines), 3):
            ticks.append(lines[i].split()[0])
        return ticks

    @cached_property
    def _x_points(self) -> list[str]:
        """X-axis positions for all k-points (as strings for exact comparison)."""
        lines = self.bands_str.splitlines()
        return [lines[i].split()[0] for i in range(self.nkpoints)]

    @cached_property
    def kticks(self) -> list[int]:
        """Indices of high-symmetry k-points."""
        x_points = np.array(self._x_points)
        tick_pos = np.array(self._tick_positions)

        ticks: list[int] = []
        nhigh_sym = len(self._high_symmetry_points) if len(self._high_symmetry_points) > 0 else len(tick_pos)

        for ihs in range(1, nhigh_sym):
            start = np.where(x_points == tick_pos[ihs - 1])[0][0]
            ticks.append(int(start))
        ticks.append(self.nkpoints - 1)
        return ticks

    @cached_property
    def ngrids(self) -> npt.NDArray[np.int64]:
        """Number of k-points per segment."""
        grids = np.diff(np.array(self.kticks))
        grids[-1] += 1
        return grids

    @cached_property
    def kpoints(self) -> npt.NDArray[np.float64]:
        """K-point coordinates as (nkpoints, 3) array."""
        if len(self._high_symmetry_points) == 0:
            return np.zeros((self.nkpoints, 3))

        x_points = np.array(self._x_points)
        tick_pos = np.array(self._tick_positions)
        kpoints = np.zeros((self.nkpoints, 3))
        nhigh_sym = len(self._high_symmetry_points)

        for ihs in range(1, nhigh_sym):
            start = np.where(x_points == tick_pos[ihs - 1])[0][0]
            end = np.where(x_points == tick_pos[ihs])[0][0] + 1
            kpoints[start:end] = np.linspace(
                self._high_symmetry_points[ihs - 1],
                self._high_symmetry_points[ihs],
                end - start,
            )
        return kpoints

    @cached_property
    def bands_hartree(self) -> npt.NDArray[np.float64]:
        """Band energies in Hartree as (nkpoints, raw_nbands) array."""
        lines = self.bands_str.splitlines()
        bands = np.zeros((self.nkpoints, self._raw_nbands))

        iline = 0
        for iband in range(self._raw_nbands):
            for ikpoint in range(self.nkpoints):
                bands[ikpoint, iband] = float(lines[iline].split()[1])
                iline += 1
            # Skip blank line between bands (only if we have k-points)
            if self.nkpoints > 0:
                iline += 1

        return bands

    @cached_property
    def bands(self) -> npt.NDArray[np.float64]:
        """Band energies in eV as (nkpoints, nbands, nspin) array.

        Note: Does NOT include Fermi energy shift - that should be applied
        by the orchestrator when creating ElectronicBandStructure.
        """
        raw_bands = self.bands_hartree * HARTREE_TO_EV
        bands = np.zeros((self.nkpoints, self.nbands, self.nspin))

        if self.nspin == 1:
            bands[:, :, 0] = raw_bands
        else:
            bands[:, :, 0] = raw_bands[:, : self.nbands]
            bands[:, :, 1] = raw_bands[:, self.nbands :]

        return bands
