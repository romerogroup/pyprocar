from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from pyprocar.utils import np_utils
from pyprocar.utils.units import RYDBERG_TO_EV


class AtomicProjXML:
    _filepath: Path
    _tree: ET.ElementTree[ET.Element]
    _root: ET.Element

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)
        self._tree = ET.parse(self._filepath)
        self._root = self._tree.getroot()

    @property
    def filepath(self) -> Path:
        return self._filepath

    @cached_property
    def root(self) -> ET.Element:
        return self._root

    @cached_property
    def tree(self) -> ET.ElementTree[ET.Element]:
        return self._tree

    @cached_property
    def n_bands(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0

        value = header_match[0].get("NUMBER_OF_BANDS")
        if value is None:
            return 0
        return int(value)

    @cached_property
    def n_kpoints(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0

        value = header_match[0].get("NUMBER_OF_K-POINTS")
        if value is None:
            return 0
        return int(value)

    @cached_property
    def n_spin_channels(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0

        value = header_match[0].get("NUMBER_OF_SPIN_COMPONENTS")
        if value is None:
            return 0
        return int(value)

    @cached_property
    def n_atm_wfc(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0

        value = header_match[0].get("NUMBER_OF_ATOMIC_WFC")
        if value is None:
            return 0
        return int(value)

    @cached_property
    def n_electrons(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0

        value = header_match[0].get("NUMBER_OF_ELECTRONS")
        if value is None:
            return 0
        return int(value)

    @cached_property
    def is_noncolinear(self) -> bool:
        header_match = self.root.findall(".//ATOMIC_SIGMA_PHI")
        return bool(header_match)

    @cached_property
    def fermi(self) -> float:
        """Fermi energy in eV"""
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0.0

        value = header_match[0].get("FERMI_ENERGY")
        if value is None:
            return 0.0
        return float(value) * RYDBERG_TO_EV

    @cached_property
    def n_spin_projections(self) -> int:
        if self.is_noncolinear:
            return 4
        return self.n_spin_channels

    @cached_property
    def eigen_states(self) -> dict[str, Any] | None:
        """
        Parses the atomic_proj.xml file and returns the bands, projections, kpoints, and weights.
         - Energies are in Rydberg.
         - Bands are subtracted by the Fermi energy.
        Returns:
            dict[str, Any] | None: A dictionary containing bands, projections, kpoints, weights.
                - bands: np.ndarray of shape (n_kpoints, n_bands, n_spin_channels) # In Rydberg
                - projections: np.ndarray of shape
                    (n_kpoints, n_bands, n_spin_projections, n_atm_wfc)
                - kpoints: np.ndarray of shape (n_kpoints, 3)
                - weights: np.ndarray of shape (n_kpoints)
        """
        eigen_states_match = self.root.findall(".//EIGENSTATES")
        if not eigen_states_match:
            return None

        eigen_state_element = eigen_states_match[0]

        kpoints_match = eigen_state_element.findall(".//K-POINT")
        bands_by_kpoint = eigen_state_element.findall(".//E")
        projections_by_kpoint = eigen_state_element.findall(".//PROJS")

        n_all_kpoints = len(bands_by_kpoint)
        raw_kpoints: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
        raw_weights: list[float] = []
        raw_bands: list[list[str]] = []
        raw_projections: list[np.ndarray[Any, np.dtype[np.complexfloating[Any, Any]]]] = []
        for i_kpoint in range(n_all_kpoints):
            band_element = bands_by_kpoint[i_kpoint]
            band_text = band_element.text
            if band_text is None:
                continue
            raw_bands.append(band_text.strip().split())

            kpoint_element = kpoints_match[i_kpoint]
            weight = float(kpoint_element.attrib["Weight"])
            raw_weights.append(weight)

            kpoint_text = kpoint_element.text
            if kpoint_text is None:
                continue
            kpoint = np.array(kpoint_text.strip().split(), dtype=float)
            raw_kpoints.append(kpoint)

            atomic_wfc_projections = projections_by_kpoint[i_kpoint].findall(".//ATOMIC_WFC")
            if not atomic_wfc_projections:
                continue
            projection = np.zeros(
                shape=(self.n_bands, self.n_atm_wfc), dtype=np_utils.COMPLEX_DTYPE
            )
            for atomic_wfc_projection in atomic_wfc_projections:
                i_atm_wfc = int(atomic_wfc_projection.attrib["index"]) - 1

                projection_text = atomic_wfc_projection.text
                if projection_text is None:
                    continue
                atomic_band_projections = projection_text.strip().split("\n")

                for i_band, atomic_band_projection in enumerate(atomic_band_projections):
                    real, imag = atomic_band_projection.strip().split()

                    projection[i_band, i_atm_wfc] += complex(float(real), float(imag))
            raw_projections.append(projection)

        raw_bands_arr = np.array(raw_bands, dtype=float)
        raw_projections_arr = np.array(raw_projections, dtype=np_utils.COMPLEX_DTYPE)
        raw_kpoints_arr = np.array(raw_kpoints, dtype=float)
        raw_weights_arr = np.array(raw_weights, dtype=float)

        bands = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin_channels), dtype=float)
        projections = np.zeros(
            shape=(self.n_kpoints, self.n_bands, self.n_spin_projections, self.n_atm_wfc),
            dtype=np_utils.COMPLEX_DTYPE,
        )

        if self.n_spin_channels == 2:
            kpoints = raw_kpoints_arr[: self.n_kpoints]
            weights = raw_weights_arr[: self.n_kpoints]
            bands[..., 0] = raw_bands_arr[: self.n_kpoints]
            bands[..., 1] = raw_bands_arr[self.n_kpoints :]

            projections[:, :, 0, :] = raw_projections_arr[: self.n_kpoints]
            projections[:, :, 1, :] = raw_projections_arr[self.n_kpoints :]
        else:
            kpoints = raw_kpoints_arr
            weights = raw_weights_arr
            bands[..., 0] = raw_bands_arr
            projections[:, :, 0, :] = raw_projections_arr

        eigen_states: dict[str, Any] = {
            "bands": bands,
            "projections": projections,
            "weights": weights,
            "kpoints": kpoints,
        }

        return eigen_states

    @cached_property
    def kpoints(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | None:
        if self.eigen_states:
            return self.eigen_states["kpoints"]
        return None

    @cached_property
    def bands(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | None:
        """"""
        if self.eigen_states:
            bands = self.eigen_states["bands"] * RYDBERG_TO_EV
            return bands
        return None

    @cached_property
    def projections(self) -> np.ndarray[Any, np.dtype[np.complexfloating[Any, Any]]] | None:
        if self.eigen_states:
            return self.eigen_states["projections"]
        return None

    @cached_property
    def weights(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | None:
        if self.eigen_states:
            return self.eigen_states["weights"]
        return None
