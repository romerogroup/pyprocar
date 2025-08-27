__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
import os
from typing import List

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator

from pyprocar.core import DensityOfStates, Structure

np.seterr(divide="ignore", invalid="ignore")

logger = logging.getLogger(__name__)


class DOSPlot:
    """
    Class to plot an electronic band structure.

    Parameters
    ----------
    dos : DensityOfStates
        An density of states pyprocar.core.DensityOfStates.
    structure : Structure
        An density of states pyprocar.core.Structure.

    ax : mpl.axes.Axes, optional
        A matplotlib Axes object. If provided the plot will be located at that ax.
        The default is None.

    Returns
    -------
    None.

    """

    def __init__(
        self,
        dos: DensityOfStates = None,
        structure: Structure = None,
        ax: mpl.axes.Axes = None,
        orientation: str = "horizontal",
        config=None,
    ):

        self.config = config

        self.dos = dos
        self.structure = structure
        self.handles = []
        self.labels = []
        self.orientation = orientation
        self.values_dict = {}

        if ax is None:
            self.fig = plt.figure(
                figsize=tuple(self.config.figure_size),
            )
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax

        if self.orientation not in ["horizontal", "vertical"]:
            raise ValueError(
                f"The orientation must be either horizontal or vertical, not {self.orientation}"
            )

        return None

    def plot_dos(self, spins: List[int] = None):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )
        energies = self.dos.energies
        dos_total = self.dos.total

        self._set_plot_limits(spin_channels)
        for ispin, spin_channel in enumerate(spin_channels):

            # flip the sign of the total dos if there are 2 spin channels
            dos_total_spin = dos_total[spin_channel, :] * (-1 if ispin > 0 else 1)
            self._plot_total_dos(energies, dos_total_spin, spin_channel)
            values_dict["energies"] = energies
            values_dict["dosTotalSpin-" + str(spin_channel)] = dos_total_spin

        self.values_dict = values_dict
        return values_dict

    def plot_parametric(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        principal_q_numbers: List[int] = [-1],
    ):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )
        dos_total, dos_total_projected, dos_projected = self._calculate_parametric_dos(
            atoms, orbitals, spin_projections, principal_q_numbers
        )

        orbital_string = ":".join([str(orbital) for orbital in orbitals])
        atom_string = ":".join([str(atom) for atom in atoms])
        spin_string = ":".join(
            [str(spin_projection) for spin_projection in spin_projections]
        )

        self._setup_colorbar(dos_projected, dos_total_projected)
        self._set_plot_limits(spin_channels)

        for ispin, spin_channel in enumerate(spin_channels):
            energies, dos_spin_total, normalized_dos_spin_projected = (
                self._prepare_parametric_spin_data(
                    spin_channel, ispin, dos_total, dos_projected, dos_total_projected
                )
            )

            self._plot_spin_data_parametric(
                energies, dos_spin_total, normalized_dos_spin_projected
            )

            if self.config.plot_total:
                self._plot_total_dos(energies, dos_spin_total, spin_channel)

            values_dict["energies"] = energies
            values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
            values_dict[
                "spinChannel-"
                + str(spin_channel)
                + f"_orbitals-{orbital_string}"
                + f"_atoms-{atom_string}"
                + f"_spinProjection-{spin_string}"
            ] = normalized_dos_spin_projected

        self.values_dict = values_dict
        return values_dict

    def plot_parametric_line(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        principal_q_numbers: List[int] = [-1],
    ):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )
        dos_total, dos_total_projected, dos_projected = self._calculate_parametric_dos(
            atoms, orbitals, spin_projections, principal_q_numbers
        )

        orbital_string = ":".join([str(orbital) for orbital in orbitals])
        atom_string = ":".join([str(atom) for atom in atoms])
        spin_string = ":".join(
            [str(spin_projection) for spin_projection in spin_projections]
        )

        self._setup_colorbar(dos_projected, dos_total_projected)
        self._set_plot_limits(spin_channels)

        for ispin, spin_channel in enumerate(spin_channels):

            energies, dos_spin_total, normalized_dos_spin_projected = (
                self._prepare_parametric_spin_data(
                    spin_channel, ispin, dos_total, dos_projected, dos_total_projected
                )
            )

            self._plot_spin_data_parametric_line(
                energies, dos_spin_total, normalized_dos_spin_projected, spin_channel
            )

            values_dict["energies"] = energies
            values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
            values_dict[
                "spinChannel-"
                + str(spin_channel)
                + f"_orbitals-{orbital_string}"
                + f"_atoms-{atom_string}"
                + f"_spinProjection-{spin_string}"
            ] = normalized_dos_spin_projected

        self.values_dict = values_dict
        return values_dict

    def plot_stack_species(
        self,
        principal_q_numbers: List[int] = [-1],
        orbitals: List[int] = None,
        spins: List[int] = None,
        overlay_mode: bool = False,
    ):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )

        orbital_label = self._get_stack_species_labels(orbitals)

        self._set_plot_limits(spin_channels)
        bottom_value = 0
        for specie in range(len(self.structure.species)):
            logger.debug(f"specie: {specie}")
            idx = np.array(self.structure.atoms) == self.structure.species[specie]
            atoms = list(np.where(idx)[0])

            orbital_string = ":".join([str(orbital) for orbital in orbitals])
            atom_string = ":".join([str(atom) for atom in atoms])
            spin_string = ":".join(
                [str(spin_projection) for spin_projection in spin_projections]
            )

            dos_total, dos_total_projected, dos_projected = (
                self._calculate_parametric_dos(
                    atoms, orbitals, spin_projections, principal_q_numbers
                )
            )
            
            logger.debug(f"dos_projected for specie {self.structure.species[specie]}: {np.mean(dos_projected)}")
            
            if np.mean(dos_projected) == 0:
                continue

            color = self.config.colors[specie]

            for ispin, spin_channel in enumerate(spin_channels):
                energies, dos_spin_total, scaled_dos_spin_projected = (
                    self._prepare_parametric_spin_data(
                        spin_channel,
                        ispin,
                        dos_total,
                        dos_projected,
                        dos_total_projected,
                        scale=True,
                    )
                )

                if overlay_mode:
                    handle = self._plot_spin_overlay(
                        energies, scaled_dos_spin_projected, spin_channel, color
                    )
                else:
                    top_value, handle = self._plot_spin_stack(
                        energies, scaled_dos_spin_projected, bottom_value, color
                    )
                    bottom_value += top_value

                label = self.structure.species[specie] + orbital_label

                values_dict["energies"] = energies
                values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
                values_dict[
                    "spinChannel-"
                    + str(spin_channel)
                    + f"_orbitals-{orbital_string}"
                    + f"_atoms-{atom_string}"
                    + f"_spinProjection-{spin_string}"
                ] = scaled_dos_spin_projected

            handle = mpatches.Patch(color=color, label=label)
            self.handles.append(handle)
            self.labels.append(label)

        if self.config.plot_total:
            total_values_dict = self.plot_dos(spin_channels)

        self.values_dict = values_dict
        return values_dict

    def plot_stack_orbitals(
        self,
        principal_q_numbers: List[int] = [-1],
        atoms: List[int] = None,
        spins: List[int] = None,
        overlay_mode: bool = False,
    ):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )

        atom_names, orb_names, orb_l = self._get_stack_orbitals_labels(atoms)

        self._set_plot_limits(spin_channels)
        bottom_value = 0
        for iorb in range(len(orb_l)):

            orbital_string = ":".join([str(orbital) for orbital in orb_l[iorb]])
            atom_string = ":".join([str(atom) for atom in atoms])
            spin_string = ":".join(
                [str(spin_projection) for spin_projection in spin_projections]
            )

            dos_total, dos_total_projected, dos_projected = (
                self._calculate_parametric_dos(
                    atoms=atoms,
                    orbitals=orb_l[iorb],
                    spin_projections=spin_projections,
                    principal_q_numbers=principal_q_numbers,
                )
            )
            
            # Skips plotting if the projected dos is zero. This is a workaround for a bug in quantum espresso stack orbitals mode
            if np.mean(dos_projected) == 0:
                continue

            color = self.config.colors[iorb]
            for ispin, spin_channel in enumerate(spin_channels):
                energies, dos_spin_total, scaled_dos_spin_projected = (
                    self._prepare_parametric_spin_data(
                        spin_channel,
                        ispin,
                        dos_total,
                        dos_projected,
                        dos_total_projected,
                        scale=True,
                    )
                )

                if overlay_mode:
                    handle = self._plot_spin_overlay(
                        energies, scaled_dos_spin_projected, spin_channel, color
                    )
                else:
                    top_value, handle = self._plot_spin_stack(
                        energies, scaled_dos_spin_projected, bottom_value, color
                    )
                    bottom_value += top_value

                label = atom_names + orb_names[iorb]  # + self.config.spin_labels[ispin]

                values_dict["energies"] = energies
                values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
                values_dict[
                    "spinChannel-"
                    + str(spin_channel)
                    + f"_orbitals-{orbital_string}"
                    + f"_atoms-{atom_string}"
                    + f"_spinProjection-{spin_string}"
                ] = scaled_dos_spin_projected

            handle = mpatches.Patch(color=color, label=label)
            self.handles.append(handle)
            self.labels.append(label)

        if self.config.plot_total:
            total_values_dict = self.plot_dos(spin_channels)

        self.values_dict = values_dict
        return values_dict

    def plot_stack(
        self,
        items: dict = None,
        principal_q_numbers: List[int] = [-1],
        spins: List[int] = None,
        overlay_mode: bool = False,
    ):
        values_dict = {}
        if len(items) is None:
            print(
                """Please provide the stacking items in which you want
                to plot, example : {'Sr':[1,2,3],'O':[4,5,6,7,8]}
                will plot the stacked plots of p orbitals of Sr and
                d orbitals of Oxygen."""
            )
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )
        self._set_plot_limits(spin_channels)
        # Defining color per specie
        counter = 0
        colors_dict = {}
        for specie in items:
            colors_dict[specie] = self.config.colors[counter]
            counter += 1

        bottom_value = 0
        for specie in items:
            idx = np.array(self.structure.atoms) == specie
            atoms = list(np.where(idx)[0])
            orbitals = items[specie]
            orbital_label = self._get_stack_labels(orbitals)

            orbital_string = ":".join([str(orbital) for orbital in orbitals])
            atom_string = ":".join([str(atom) for atom in atoms])
            spin_string = ":".join(
                [str(spin_projection) for spin_projection in spin_projections]
            )

            dos_total, dos_total_projected, dos_projected = (
                self._calculate_parametric_dos(
                    atoms=atoms,
                    orbitals=orbitals,
                    spin_projections=spin_projections,
                    principal_q_numbers=principal_q_numbers,
                )
            )
            
            logger.debug(f"dos_projected: {np.mean(dos_projected)}")
            
            if np.mean(dos_projected) == 0:
                continue

            color = colors_dict[specie]
            for ispin, spin_channel in enumerate(spin_channels):
                energies, dos_spin_total, scaled_dos_spin_projected = (
                    self._prepare_parametric_spin_data(
                        spin_channel,
                        ispin,
                        dos_total,
                        dos_projected,
                        dos_total_projected,
                        scale=True,
                    )
                )

                if overlay_mode:
                    handle = self._plot_spin_overlay(
                        energies, scaled_dos_spin_projected, spin_channel, color
                    )
                else:
                    top_value, handle = self._plot_spin_stack(
                        energies, scaled_dos_spin_projected, bottom_value, color
                    )
                    bottom_value += top_value

                label = specie + orbital_label
                values_dict["energies"] = energies
                values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
                values_dict[
                    "spinChannel-"
                    + str(spin_channel)
                    + f"_orbitals-{orbital_string}"
                    + f"_atoms-{atom_string}"
                    + f"_spinProjection-{spin_string}"
                ] = scaled_dos_spin_projected

            handle = mpatches.Patch(color=color, label=label)
            self.handles.append(handle)
            self.labels.append(label)

        if self.config.plot_total:
            total_values_dict = self.plot_dos(spin_channels)

        self.values_dict = values_dict

        return values_dict

    def _calculate_parametric_dos(
        self, atoms, orbitals, spin_projections, principal_q_numbers
    ):
        dos_total = np.array(self.dos.total)
        if self.dos.n_spins == 4:
            dos_total_projected = self.dos.dos_sum(spins=spin_projections)
        else:
            dos_total_projected = self.dos.dos_sum()
        dos_projected = self.dos.dos_sum(
            atoms=atoms,
            principal_q_numbers=principal_q_numbers,
            orbitals=orbitals,
            spins=spin_projections,
        )
            
        return dos_total, dos_total_projected, dos_projected

    def _get_spins_projections_and_channels(self, spins):
        """
        This function determines the spin channels and projections from the spins keywrod argument.

        Parameters
        ----------
        spins : list of int, optional
            A list of spins, by default None

        Returns
        -------
        spin_projections : list of int
            A list of spin projections
        spin_channels : list of int
            A list of spin channels
        """

        if self.dos.is_non_collinear:
            spin_projections = spins if spins else [0, 1, 2]
            spin_channels = [0]
        else:
            spin_channel_list = range(self.dos.n_spins)
            spin_projections = spins if spins else spin_channel_list
            spin_channels = spins if spins else spin_channel_list

        return spin_projections, spin_channels

    def _get_stack_species_labels(self, orbitals):
        # This condition will depend on which orbital basis is being used.
        if (
            self.dos.is_non_collinear
            and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8
        ):
            spins = [0]
            if orbitals:
                print("The plot only considers orbitals", orbitals)
                label = "-"
                if sum([x in orbitals for x in [0, 1]]) == 2:
                    label += "s-j=0.5"
                if sum([x in orbitals for x in [2, 3]]) == 2:
                    label += "p-j=0.5"
                if sum([x in orbitals for x in [4, 5, 6, 7]]) == 4:
                    label += "p-j=1.5"
                if sum([x in orbitals for x in [8, 9, 10, 11]]) == 4:
                    label += "d-j=1.5"
                if sum([x in orbitals for x in [12, 13, 14, 15, 16, 17]]) == 6:
                    label += "d-j=2.5"
                if sum([x in orbitals for x in [18, 19, 20, 21, 22, 23]]) == 6:
                    label += "f-j=2.5"
                if sum([x in orbitals for x in [24, 25, 26, 27, 28, 29, 30, 31]]) == 8:
                    label += "f-j=3.5"
            else:
                if len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8:
                    label = "-spdf-j=0.5,1.5,2.5,3.5"
                else:
                    label = "-"
        else:
            if orbitals:
                print("The plot only considers orbitals", orbitals)
                label = "-"
                if sum([x in orbitals for x in [0]]) == 1:
                    label += "s"
                if sum([x in orbitals for x in [1, 2, 3]]) == 3:
                    label += "p"
                if sum([x in orbitals for x in [4, 5, 6, 7, 8]]) == 5:
                    label += "d"
                if sum([x in orbitals for x in [9, 10, 11, 12, 13, 14, 15]]) == 7:
                    label += "f"
            else:
                if len(self.dos.projected[0][0]) == 1 + 3 + 5:
                    label = "-spd"
                elif len(self.dos.projected[0][0]) == 1 + 3 + 5 + 7:
                    label = "-spdf"
                else:
                    label = "-"
        return label

    def _get_stack_orbitals_labels(self, atoms):
        atom_names = ""
        if atoms:
            atom_names = ""
            for ispc in np.unique(np.array(self.structure.atoms)[atoms]):
                atom_names += ispc + "-"
        all_atoms = ""
        for ispc in np.unique(np.array(self.structure.atoms)):
            all_atoms += ispc + "-"
        if atom_names == all_atoms:
            atom_names = ""

        logger.debug(f"self.dos.projected[0][0]: {len(self.dos.projected[0][0])}")
        logger.debug(f"self.dos.is_non_collinear: {self.dos.is_non_collinear}")
        
        if (
            self.dos.is_non_collinear
            and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8
        ):
            orb_names = ["s-j=0.5", "p-j=0.5", "p-j=1.5", "d-j=1.5", "d-j=2.5", "f-j=2.5", "f-j=3.5"]
            orb_l = [
                [0, 1],
                [2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31]
            ]
        elif len(self.dos.projected[0][0]) == 1 + 3 + 5:
            orb_names = ["s", "p", "d"]
            orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]
        elif len(self.dos.projected[0][0]) == 1 + 3 + 5 + 7:
            orb_names = ["s", "p", "d", "f"]
            orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]]

        return atom_names, orb_names, orb_l

    def _get_stack_labels(self, orbitals):
        if (
            self.dos.is_non_collinear
            and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8
        ):
            if len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8:
                all_orbitals = "-spdf-j=0.5,1.5,2.5,3.5"
            else:
                all_orbitals = "-"
        else:
            if len(self.dos.projected[0][0]) == (1 + 3 + 5):
                all_orbitals = "spd"
            elif len(self.dos.projected[0][0]) == (1 + 3 + 5 + 7):
                all_orbitals = "spdf"
            else:
                all_orbitals = ""

        label = "-"
        # For coupled basis
        if len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8:
            if sum([x in orbitals for x in [0, 1]]) == 2:
                label += "s-j=0.5"
            if sum([x in orbitals for x in [2, 3]]) == 2:
                label += "p-j=0.5"
            if sum([x in orbitals for x in [4, 5, 6, 7]]) == 4:
                label += "p-j=1.5"
            if sum([x in orbitals for x in [8, 9, 10, 11]]) == 4:
                label += "d-j=1.5"
            if sum([x in orbitals for x in [12, 13, 14, 15, 16, 17]]) == 6:
                label += "d-j=2.5"
            if sum([x in orbitals for x in [18, 19, 20, 21, 22, 23]]) == 6:
                label += "f-j=2.5"
            if sum([x in orbitals for x in [24, 25, 26, 27, 28, 29, 30, 31]]) == 8:
                label += "f-j=3.5"
            if label == "-" + all_orbitals:
                label = ""
        # For uncoupled basis
        else:
            if sum([x in orbitals for x in [0]]) == 1:
                label += "s"
            if sum([x in orbitals for x in [1, 2, 3]]) == 3:
                label += "p"
            if sum([x in orbitals for x in [4, 5, 6, 7, 8]]) == 5:
                label += "d"
            if sum([x in orbitals for x in [9, 10, 11, 12, 13, 14, 15]]) == 7:
                label += "f"
            if label == "-" + all_orbitals:
                label = ""
        return label

    def _setup_colorbar(self, dos_projected, dos_total_projected):

        vmin, vmax = self._get_color_limits(dos_projected, dos_total_projected)
        cmap = mpl.cm.get_cmap(self.config.cmap)

        if self.config.plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax
            )
            cb.ax.tick_params(labelsize=self.config.colorbar_tick_labelsize)
            cb.set_label(
                self.config.colorbar_title,
                size=self.config.colorbar_title_size,
                rotation=270,
                labelpad=self.config.colorbar_title_padding,
            )

    def _get_color_limits(self, dos_projected, dos_total_projected):
        if self.config.clim:
            self.clim = self.config.clim
        else:
            self.clim = [0, 0]
            self.clim[0] = dos_projected.min() / dos_total_projected.max()
            self.clim[1] = dos_projected.max() / dos_total_projected.max()
        return self.clim

    def _set_plot_limits(self, spin_channels):
        total_max = 0
        for ispin in range(len(spin_channels)):
            tmp_max = self.dos.total[ispin].max()
            if tmp_max > total_max:
                total_max = tmp_max

        if self.orientation == "horizontal":
            x_label = self.config.x_label
            y_label = self.config.y_label
            xlim = [self.dos.energies.min(), self.dos.energies.max()]
            ylim = (
                [-self.dos.total.max(), total_max]
                if len(spin_channels) == 2
                else [0, total_max]
            )
        elif self.orientation == "vertical":
            x_label = self.config.y_label
            y_label = self.config.x_label
            xlim = (
                [-self.dos.total.max(), total_max]
                if len(spin_channels) == 2
                else [0, total_max]
            )
            ylim = [self.dos.energies.min(), self.dos.energies.max()]

        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def _prepare_parametric_spin_data(
        self,
        spin_channel,
        ispin,
        dos_total,
        dos_projected,
        dos_total_projected,
        scale=False,
    ):
        """
        Prepares the data for the parametric plot.

        Parameters
        ----------
        spin_channel : int
            The spin channel being plotted
        ispin : int
            The index of the spin channel being plotted
        dos_total : np.ndarray
            The total density of states
        dos_projected : np.ndarray
            The projected density of states
        dos_total_projected : np.ndarray
            The projected total density of states
        scale : bool, optional
            Boolean to scale the projected density of states

        Returns
        -------
        x : np.ndarray
            The x values
        y_total : np.ndarray
            The total y values
        y_projected : np.ndarray
            The projected y values

        """
        energies = self.dos.energies
        dos_total = dos_total[spin_channel, :]
        dos_projected = dos_projected[spin_channel, :]
        dos_total_projected = dos_total_projected[spin_channel, :]

        # Should be between 0 and 1
        normalized_dos_projected = dos_projected / dos_total_projected
        
        # assert normalized_dos_projected.min() >= 0 and normalized_dos_projected.max() <= 1, "Issue with the normalization of the projected DOS"
        # Removing issues points due to divisions by zero
        normalized_dos_projected = np.nan_to_num(normalized_dos_projected, 0)
        threshold = max(abs(dos_total)) + 1
        normalized_dos_projected[np.abs(normalized_dos_projected) > threshold] = 0
        
    
        if ispin > 0 and len(self.dos.total) > 1:
            dos_total *= -1
            dos_projected *= -1
            dos_total_projected *= -1

        if scale:
            scaled_dos_projected = normalized_dos_projected * dos_total
            final_dos_projected = scaled_dos_projected
            threshold = max(abs(dos_total)) + 1
            final_dos_projected[np.abs(final_dos_projected) > threshold] = 0
        else:
            final_dos_projected = normalized_dos_projected

        return energies, dos_total, final_dos_projected

    def _get_bar_color(self, values):
        cmap = mpl.cm.get_cmap(self.config.cmap)
        return [cmap(value) for value in values]

    def _set_data_to_orientation(self, energies, dos_total):
        if self.orientation == "horizontal":
            data = {
                "x": energies,
                "y": dos_total,
                "energies": energies,
                "dos_value": dos_total,
                "xlim": [energies.min(), energies.max()],
                "ylim": [dos_total.min(), dos_total.max()],
                "xlabel": self.config.x_label,
                "ylabel": self.config.y_label,
                "fill_func": self.ax.fill_between,
            }
        elif self.orientation == "vertical":
            data = {
                "x": dos_total,
                "y": energies,
                "energies": energies,
                "dos_value": dos_total,
                "xlim": [dos_total.min(), dos_total.max()],
                "ylim": [energies.min(), energies.max()],
                "xlabel": self.config.y_label,
                "ylabel": self.config.x_label,
                "fill_func": self.ax.fill_betweenx,
            }
        return data

    def _plot_total_dos(self, energies, dos_total_spin, spin_channel):
        """
        Plots the total DOS.

        Parameters
        ----------
        spin_channel : int
            The spin channel being plotted
        spins_index : int
            The index of the spin channels being plotted. If spin index is 1,
            then the spins dos is inverted on the axis.

        Returns
        -------
        None
            None
        """

        data = self._set_data_to_orientation(energies, dos_total_spin)

        self.ax.plot(
            data["x"],
            data["y"],
            color=self.config.color,
            alpha=self.config.opacity[spin_channel],
            linestyle=self.config.linestyle[spin_channel],
            label=self.config.spin_labels[spin_channel],
            linewidth=self.config.linewidth[spin_channel],
        )

    def _plot_spin_data_parametric(self, energies, dos_total, normalized_dos_projected):
        bar_color = self._get_bar_color(normalized_dos_projected)
        data = self._set_data_to_orientation(energies, dos_total)

        self._plot_fill_between(
            x=data["energies"],
            y=data["dos_value"],
            fill_func=data["fill_func"],
            bar_color=bar_color,
        )

    def _plot_spin_data_parametric_line(
        self, energies, dos_total_spin, normalized_dos_spin_projected, spin_channel
    ):

        data = self._set_data_to_orientation(energies, dos_total_spin)
        points = np.array([data["x"], data["y"]]).T.reshape(-1, 1, 2)

        # generates line segments. This is the reason for the offset of the points
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = mpl.colors.Normalize(vmin=self.clim[0], vmax=self.clim[1])
        lc = LineCollection(segments, cmap=plt.get_cmap(self.config.cmap), norm=norm)
        lc.set_array(normalized_dos_spin_projected)
        lc.set_linewidth(self.config.linewidth[spin_channel])
        lc.set_linestyle(self.config.linestyle[spin_channel])

        handle = self.ax.add_collection(lc)
        self.handles.append(handle)

    def _plot_fill_between(
        self, x, y, fill_func, bottom_value=0, bar_color=None, color=None
    ):
        if color:
            final_color = color
            handle = fill_func(x, y + bottom_value, bottom_value, color=final_color)
        if bar_color:
            for i in range(len(x) - 1):
                handle = fill_func(
                    [x[i], x[i + 1]], [y[i], y[i + 1]], color=bar_color[i]
                )
        return handle

    def _plot_spin_stack(
        self, energies, scaled_projected_dos, bottom_value=0, color=None
    ):

        data = self._set_data_to_orientation(energies, scaled_projected_dos)
        handle = self._plot_fill_between(
            x=data["energies"],
            y=data["dos_value"],
            fill_func=data["fill_func"],
            bottom_value=bottom_value,
            color=color,
        )

        bottom_value = data["dos_value"]
        return bottom_value, handle

    def _plot_spin_overlay(
        self, energies, scaled_projected_dos, spin_channel, color=None
    ):

        data = self._set_data_to_orientation(energies, scaled_projected_dos)

        (handle,) = self.ax.plot(
            data["x"],
            data["y"],
            color=color,
            alpha=self.config.opacity[spin_channel],
            linestyle=self.config.linestyle[spin_channel],
            label=self.config.spin_labels[spin_channel],
            linewidth=self.config.linewidth[spin_channel],
        )

        return handle

    def set_xticks(
        self, tick_positions: List[int] = None, tick_names: List[str] = None
    ):
        """A method to set the xticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """

        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        if self.config.major_x_tick_params:
            self.ax.tick_params(**self.config.major_x_tick_params)
        if self.config.minor_x_tick_params:
            self.ax.tick_params(**self.config.minor_x_tick_params)
        return None

    def set_yticks(
        self, tick_positions: List[int] = None, tick_names: List[str] = None
    ):
        """A method to set the yticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """
        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)

        if self.config.major_y_tick_params:
            self.ax.tick_params(**self.config.major_y_tick_params)
        if self.config.minor_y_tick_params:
            self.ax.tick_params(**self.config.minor_y_tick_params)
        return None

    def set_xlim(self, interval: List[int] = None):
        """A method to set the xlim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The x interval, by default None
        """
        if interval is not None:
            self.ax.set_xlim(interval)
        return None

    def set_ylim(self, interval: List[int] = None):
        """A method to set the ylim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The y interval, by default None
        """
        if interval is not None:
            self.ax.set_ylim(interval)

        return None

    def set_xlabel(self, label: str):
        """A method to set the x label

        Parameters
        ----------
        label : str
            The x label name

        Returns
        -------
        None
            None
        """
        if self.config.x_label:
            self.ax.set_xlabel(self.config.x_label, **self.config.x_label_params)
        else:
            self.ax.set_xlabel(label, **self.config.x_label_params)
        return None

    def set_ylabel(self, label: str):
        """A method to set the y label

        Parameters
        ----------
        label : str
            The y label name

        Returns
        -------
        None
            None
        """
        if self.config.y_label:
            self.ax.set_ylabel(self.config.y_label, **self.config.y_label_params)
        else:
            self.ax.set_ylabel(label, **self.config.y_label_params)

    def legend(self, labels: List[str] = None):
        """A method to include the legend

        Parameters
        ----------
        label : str
            The labels for the legend

        Returns
        -------
        None
            None
        """
        if labels == None:
            labels = self.labels
        if self.config.legend and len(labels) != 0:
            if len(self.handles) != len(labels):
                raise ValueError(
                    f"The number of labels and handles should be the same, currently there are {len(self.handles)} handles and {len(labels)} labels"
                )
            self.ax.legend(self.handles, labels, **self.config.legend_params)
        return None

    def draw_fermi(self, value, orientation: str = "horizontal"):
        """A method to draw the fermi surface

        Parameters
        ----------
        orientation : str, optional
            Boolean to plot vertical or horizontal, by default 'horizontal'
        color : str, optional
            A color , by default "blue"
        linestyle : str, optional
            THe line style, by default "dotted"
        linewidth : float, optional
            The linewidth, by default 1

        Returns
        -------
        None
            None
        """
        if orientation == "horizontal":
            self.ax.axvline(
                x=value,
                color=self.config.fermi_color,
                linestyle=self.config.fermi_linestyle,
                linewidth=self.config.fermi_linewidth,
            )
        elif orientation == "vertical":
            self.ax.axhline(
                y=value,
                color=self.config.fermi_color,
                linestyle=self.config.fermi_linestyle,
                linewidth=self.config.fermi_linewidth,
            )
        return None

    def draw_baseline(self, value, orientation: str = "horizontal"):
        """A method to draw the baseline

        Parameters
        ----------
        value : float
            The value of the baseline
        """
        if orientation == "horizontal":
            self.ax.axhline(y=value, **self.config.baseline_params)
        elif orientation == "vertical":
            self.ax.axvline(x=value, **self.config.baseline_params)
        return None

    def grid(self):
        """A method to include a grid on the plot.

        Returns
        -------
        None
            None
        """
        if self.config.grid:
            self.ax.grid(
                self.config.grid,
                which=self.config.grid_which,
                color=self.config.grid_color,
                linestyle=self.config.grid_linestyle,
                linewidth=self.config.grid_linewidth,
            )
        return None

    def show(self):
        """A method to show the plot

        Returns
        -------
        None
            None
        """
        plt.show()
        return None

    def save(self, filename: str = "dos.pdf"):
        """A method to save the plot

        Parameters
        ----------
        filename : str, optional
            The filename, by default 'dos.pdf'

        Returns
        -------
        None
            None
        """

        plt.savefig(filename, dpi=self.config.dpi, bbox_inches="tight")
        plt.clf()
        return None

    def update_config(self, config_dict):
        for key, value in config_dict.items():
            self.config[key]["value"] = value

    def export_data(self, filename):
        """
        This method will export the data to a csv file

        Parameters
        ----------
        filename : str
            The file name to export the data to

        Returns
        -------
        None
            None
        """
        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"The file type must be {possible_file_types}")
        if self.values_dict is None:
            raise ValueError("The data has not been plotted yet")

        column_names = list(self.values_dict.keys())
        sorted_column_names = [None] * len(column_names)
        index = 0
        for column_name in column_names:
            if "energies" in column_name.split("_")[0]:
                sorted_column_names[index] = column_name
                index += 1

        for column_name in column_names:
            if "dosTotalSpin" in column_name.split("_")[0]:
                sorted_column_names[index] = column_name
                index += 1
        for ispin in range(2):
            for column_name in column_names:

                if "spinChannel-0" in column_name.split("_")[0] and ispin == 0:
                    sorted_column_names[index] = column_name
                    index += 1
                if "spinChannel-1" in column_name.split("_")[0] and ispin == 1:
                    sorted_column_names[index] = column_name
                    index += 1

        column_names.sort()
        if file_type == "csv":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, index=False)
        elif file_type == "txt":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep="\t", index=False)
        elif file_type == "json":
            with open(filename, "w") as outfile:
                for key, value in self.values_dict.items():
                    self.values_dict[key] = value.tolist()
                json.dump(self.values_dict, outfile)
        elif file_type == "dat":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep=" ", index=False)
