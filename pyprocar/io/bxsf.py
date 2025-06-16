__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import re
from pathlib import Path
from typing import List, Union

import numpy as np

from pyprocar.core import ElectronicBandStructure


class BxsfParser:
    """
    The class is used to parse the information inside bxsf files

    Parameters
    ----------
    filepaths : list, optional
        This is a list of .bxsf filenames to parse through.
        It is a list because in some codes there can be multiple .bsxf files representing spin-up and spin-sown bands
        ,by default ["in.bxsf"]

    """

    def __init__(self, filepaths: Union[list[Path], Path] = Path("in.bxsf")):

        self.reciprocal_lattice = None
        self.origin = None

        self.nkfs_dim = None
        self.nkfs = None

        self.nk_dim = None
        self.nk = None
        self.kpoints = None

        self.band_labels = None
        self.n_bands = None
        self.bands = None

        self.parse_bxsf(filepaths=filepaths)

        self.ebs = ElectronicBandStructure(
            kpoints=self.kpoints,
            bands=self.bands,
            projected=None,
            efermi=self.e_fermi,
            kpath=None,
            projected_phase=None,
            labels=None,
            reciprocal_lattice=self.reciprocal_lattice,
            interpolation_factor=None,
        )
        return None

    def parse_bxsf(self, filepaths: Union[list[Path], Path]):
        """A Helper method to parse bxsf files

        Parameters
        ----------
        infiles : List
            This is a list of .bxsf filenames to parse through.
        """

        band_labels = []
        # If 2 bxsf files search for total number of bands in both files
        for ispin, filepath in enumerate(filepaths):
            with open(filepath, "r") as f:
                data = f.read()
            band_labels_spin = re.findall("BAND\:\s*(.*)", data)

            band_labels_spin = [int(band_label) for band_label in band_labels_spin]
            raw_nkfs = re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n(.*)", data)[
                0
            ]
            self.nkfs_dim = np.array([int(x) for x in raw_nkfs.split()])
            self.nkfs = np.product(self.nkfs_dim)

            band_labels.extend(band_labels_spin)

        self.n_bands = max(band_labels)
        self.bands = np.zeros(
            shape=[
                self.nkfs_dim[0] * self.nkfs_dim[1] * self.nkfs_dim[2],
                self.n_bands,
                2,
            ]
        )

        # populates bands array and kpoints
        for ispin, filepath in enumerate(filepaths):
            with open(filepath, "r") as f:
                data = f.read()

            self.e_fermi = float(re.findall("Fermi\sEnergy:\s*([\d.]*)", data)[0])

            self.origin = re.findall(
                "BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n.*\n(.*)", data
            )[0].split()
            self.origin = np.array([float(x) for x in self.origin])

            self.reciprocal_lattice = re.findall(
                "BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n.*\n.*\n" + 3 * "\s*(.*)\s*\n",
                data,
            )[0]
            self.reciprocal_lattice = np.array(
                [[float(y) for y in x.split()] for x in self.reciprocal_lattice]
            )

            # Bxsf format adds extra redundant +1 dimension-size nkfs is including this dimension
            raw_nkfs = re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n(.*)", data)[
                0
            ]
            self.nkfs_dim = np.array([int(x) for x in raw_nkfs.split()])
            self.nkfs = np.product(self.nkfs_dim)

            # Bxsf format adds extra redundant +1 dimension-size, nk is excluding this dimension
            self.nk_dim = np.array([int(x) - 1 for x in raw_nkfs.split()])
            self.nk = np.product(self.nk_dim)
            self.band_labels = re.findall("BAND\:\s*(.*)", data)

            # Number of bands
            self.n_bands = int(
                re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n\s*(\d*)", data)[0]
            )

            band_labels = re.findall("BAND\:\s*(.*)", data)
            band_labels = [int(band_label) for band_label in band_labels_spin]

            if ispin == 0:
                self.kpoints = np.zeros(
                    shape=[self.nkfs_dim[0] * self.nkfs_dim[1] * self.nkfs_dim[2], 3]
                )
                # 2 for spin

            band_blocks = re.findall("(?<=BAND:).*\n([\s\S]*?)(?=[A-Za-z])", data)
            for i, (band_label, band_block) in enumerate(zip(band_labels, band_blocks)):
                band_energies = band_block.split()
                band_energies = [float(energy) for energy in band_energies]
                i_kpoint = 0

                iband = band_label - 1
                extra_band_energy_indices = []
                for i in range(self.nkfs_dim[0]):
                    for j in range(self.nkfs_dim[1]):
                        for k in range(self.nkfs_dim[2]):

                            self.bands[i_kpoint, iband, ispin] = band_energies[i_kpoint]
                            self.kpoints[i_kpoint, :] = np.array(
                                [
                                    (i) / (self.nkfs_dim[0] - 1),
                                    (j) / (self.nkfs_dim[1] - 1),
                                    (k) / (self.nkfs_dim[2] - 1),
                                ]
                            )
                            if (
                                i == self.nkfs_dim[0] - 1
                                or j == self.nkfs_dim[1] - 1
                                or k == self.nkfs_dim[2] - 1
                            ):
                                extra_band_energy_indices.append(i_kpoint)

                            i_kpoint += 1

        # Deletes extra kpoints
        self.kpoints = np.delete(self.kpoints, extra_band_energy_indices, axis=0)
        # self.kpoints = np.around(self.kpoints.dot(np.linalg.inv(self.reciprocal_lattice)),decimals=8)

        # Deletes extra band energies
        self.bands = np.delete(self.bands, extra_band_energy_indices, axis=0)

        return None
