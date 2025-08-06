import copy
import logging
import os
from typing import List

import numpy as np
import pyvista as pv
import vtk
from matplotlib import cm
from matplotlib import colors as mpcolors
from PIL import Image
from pyvista.core.filters import _get_output
from pyvista.core.utilities import (
    NORMALS,
    assert_empty_kwargs,
    generate_plane,
    get_array,
    get_array_association,
    try_callback,
)
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    crinkle_algorithm,
    outline_algorithm,
    pointset_to_polydata_algorithm,
    set_algorithm_input,
)

from pyprocar.core.fermisurface3D import FermiSurface3D
from pyprocar.utils import ROOT

logger = logging.getLogger(__name__)

# TODO: Decouple FermiDataHandler from FermiVisualizer
# TODO: Normalize data does not work.


class FermiDataHandler:

    def __init__(self, ebs, config):
        self.config = config
        self.initial_ebs = copy.copy(ebs)
        self.ebs = ebs

    def _determine_spin_projections(self, spins):
        """
        This method will determine spin projections.

        Parameters
        ----------
        spins : List[int], optional
            List of spins, by default None

        Returns
        -------
        spins : List[int]
            List of spins
        """
        logger.info("___Determining spin projections___")
        logger.info(f"Initial spins: {spins}")
        if spins is None:
            if self.initial_ebs.bands.shape[2] == 1 or np.all(
                self.initial_ebs.bands[:, :, 1] == 0
            ):
                spins = [0]
            else:
                spins = [0, 1]

        if self.initial_ebs.nspins == 2 and spins is None:
            spin_pol = [0, 1]
        elif self.initial_ebs.nspins == 2:
            spin_pol = spins
        else:
            spin_pol = [0]

        logger.info(f"Final spins: {spins}.")
        logger.info(f"Final spin_pol: {spin_pol}. This is intented ")
        logger.info(f"____End of spin projections processing___")
        return spins, spin_pol

    def _process_data_for_parametric_mode(self, spins, atoms, orbitals, bands_to_keep):
        logger.info("___Processing data for parametric mode___")
        spd = []
        if orbitals is None and self.initial_ebs.projected is not None:
            orbitals = np.arange(self.initial_ebs.norbitals, dtype=int)
        if atoms is None and self.initial_ebs.projected is not None:
            atoms = np.arange(self.initial_ebs.natoms, dtype=int)

        if self.initial_ebs.is_non_collinear:
            projected = self.initial_ebs.ebs_sum(
                spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=True
            )
        else:
            projected = self.initial_ebs.ebs_sum(
                spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False
            )

        for ispin in self.spin_pol:
            spin_bands_projections = []
            for iband in bands_to_keep:
                spin_bands_projections.append(projected[:, iband, ispin])
            spd.append(spin_bands_projections)
        spd = np.array(spd).T
        spins = np.arange(spd.shape[2])
        logger.info(f"Orbitals considered: {orbitals}")
        logger.info(f"Atoms considered: {atoms}")
        logger.info(f"Final spins: {spins}")
        logger.info(f"spd after summing over the projections shape: {spd.shape}")
        logger.info(f"First kpoint and bands: {spd[0,0,:]}")
        logger.info(f"____End of parametric mode processing___")
        return spd, spins

    def _determine_bands_near_fermi(self, fermi_tolerance):
        logger.info("___Determining bands near fermi___")

        band_near_fermi = []
        for iband in range(len(self.initial_ebs.bands[0, :, 0])):
            fermi_surface_test = len(
                np.where(
                    np.logical_and(
                        self.initial_ebs.bands[:, iband, 0]
                        >= self.initial_ebs.efermi - fermi_tolerance,
                        self.initial_ebs.bands[:, iband, 0]
                        <= self.initial_ebs.efermi + fermi_tolerance,
                    )
                )[0]
            )
            if fermi_surface_test != 0:
                band_near_fermi.append(iband)

        logger.info(f"Bands Near Fermi : {band_near_fermi}")
        logger.info(f"____End of bands near fermi processing___")
        return band_near_fermi

    def _process_spd(self, mode, spins, atoms, orbitals, bands_to_keep):
        logger.info("___Processing spd___")
        if mode == "parametric":
            spd, spins = self._process_data_for_parametric_mode(
                spins, atoms, orbitals, bands_to_keep
            )
        else:
            spd = np.zeros(
                shape=(self.initial_ebs.nkpoints, len(bands_to_keep), len(spins))
            )
        logger.debug(f"spd after summing over the projections shape: {spd.shape}")
        logger.info(f"____End of spd processing___")
        return spd

    def _process_spd_spin_texture(
        self, spin_texture, spins, atoms, orbitals, bands_to_keep
    ):
        logger.info("___Processing spd_spin_texture___")
        spd_spin = []
        if spin_texture:
            ebs = copy.deepcopy(self.initial_ebs)
            ebs.projected = ebs.ebs_sum(
                spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False
            )

            for iband in bands_to_keep:
                spd_spin.append(
                    [
                        ebs.projected[:, iband, [1]],
                        ebs.projected[:, iband, [2]],
                        ebs.projected[:, iband, [3]],
                    ]
                )
            spd_spin = np.array(spd_spin)[:, :, :, 0]
            spd_spin = np.swapaxes(spd_spin, 0, 1)
            spd_spin = np.swapaxes(spd_spin, 0, 2)
            logger.debug(
                f"spd_spin after summing over the projections shape: {spd_spin.shape}"
            )
        else:
            for iband in bands_to_keep:
                spd_spin.append(None)
        if len(spd_spin) == 0:
            spd_spin = None

        logger.info(f"____End of spd_spin_texture processing___")
        return spd_spin

    def _initialize_properties(self, property_name=None):
        logger.info("___Initializing properties___")
        if property_name:
            current_emplemented_properties = [
                "fermi_velocity",
                "fermi_speed",
                "avg_inv_effective_mass",
            ]
            if property_name not in current_emplemented_properties:
                tmp = f"You must choose one of the following properies : {current_emplemented_properties}"
                raise ValueError(tmp)
            else:
                if property_name == "fermi_velocity":
                    self.ebs.fermi_velocity
                elif property_name == "fermi_speed":
                    self.ebs.fermi_speed
                elif property_name == "avg_inv_effective_mass":
                    self.ebs.avg_inv_effective_mass

    def _get_spin_pol_indices(self, fermi_surfaces):
        """
        This method will return the spin and band indices
        when combining fermi surfaces of different spins

        Parameters
        ----------
        fermi_surfaces : List[pyprocar.core.FermiSurface3D]
            The list of fermi surfaces

        Returns
        -------
        spins_band_index : List[int]
            The spin and band indices for the fermi surfaces
        spins_index : List[int]
            The spin indices for the fermi surfaces
        """
        spins_band_index = []
        spins_index = []
        # Iterate over the fermi surfaces
        spin_band_index = []
        for ispin, surface in enumerate(fermi_surfaces):
            if surface.points.shape[0] == 0:
                logger.debug(
                    f"No Fermi surface found for spin {ispin}. Skipping adding spin_band_index "
                )
                continue

            if spin_band_index is None:
                spin_band_index = list(surface.point_data["band_index"])
            else:
                spin_band_index = list(
                    surface.point_data["band_index"] + len(np.unique(spin_band_index))
                )

            # Reverse the band indices to ensure the ordering is correct.
            # PyVista prepenmds the data to point data
            spin_band_index.reverse()
            spins_band_index.extend(spin_band_index)

            spin_index = [ispin] * len(surface.points)

            spin_index.reverse()
            spins_index.extend(spin_index)

        # Reverse the spin indices to ensure the ordering is correct.

        spins_index.reverse()
        spins_band_index.reverse()
        return spins_band_index, spins_index

    def _merge_fermi_surfaces(self, fermi_surfaces):
        logger.info(f"____ Merging Fermi Surfaces of different spins ____")
        # Gets the spin-band and spin indices for the the combines fermi surfaces
        spins_band_index, spins_index = self._get_spin_pol_indices(fermi_surfaces)
        fermi_surface = None
        for i, surface in enumerate(fermi_surfaces):
            if surface.points.shape[0] == 0:
                logger.debug(f"No Fermi surface found for spin {i}. Skipping merging.")
                continue
            if fermi_surface is None:
                fermi_surface = surface
            else:
                fermi_surface.merge(surface, merge_points=False, inplace=True)

        fermi_surface.point_data["spin_index"] = np.array(spins_index)
        return fermi_surface

    def process_data(
        self,
        bands: List[int] = None,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        spin_texture: bool = False,
        fermi_tolerance: float = 0.1,
    ):
        """A helper method to process/aggregate data

        Parameters
        ----------
        bands : List[int], optional
            List of bands, by default None
        atoms : List[int], optional
            List of stoms, by default None
        orbitals : List[int], optional
            List of orbitals, by default None
        spins : List[int], optional
            List of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        fermi_tolerance : float, optional
            The tolerace to search for bands around the fermi energy, by default 0.1

        Returns
        -------
        _type_
            _description_
        """
        logger.info("___Processing data fermi surface___")

        bands_to_keep = bands
        if bands_to_keep is None:
            bands_to_keep = np.arange(len(self.initial_ebs.bands[0, :, 0]))

        self.band_near_fermi = self._determine_bands_near_fermi(
            self.config.fermi_tolerance
        )
        spins, self.spin_pol = self._determine_spin_projections(spins)

        spd = self._process_spd(self.config.mode, spins, atoms, orbitals, bands_to_keep)
        spd_spin = self._process_spd_spin_texture(
            spin_texture, spins, atoms, orbitals, bands_to_keep
        )
        self.spd = spd
        self.spd_spin = spd_spin
        self.bands_to_keep = bands_to_keep
        self.spins = spins
        return spd, spd_spin, bands_to_keep, spins

    def get_surface_data(
        self, property_name=None, fermi: float = None, fermi_shift: float = 0.0
    ):
        logger.info(f"____ Getting Fermi Surface Data ____")
        if self.config.mode is None:
            raise "You must call process data function before get_surface"

        self._initialize_properties(property_name)

        fermi_surfaces = []
        for ispin, spin in enumerate(self.spin_pol):
            ebs = copy.copy(self.ebs)
            ebs.properties_from_scratch = True
            ebs.bands = ebs.bands[:, self.bands_to_keep, spin]

            logger.debug(f"Bands shape for spin {spin}: {ebs.bands.shape}")

            fermi_surface3D = FermiSurface3D(
                ebs=ebs,
                fermi=fermi,
                fermi_shift=fermi_shift,
                interpolation_factor=self.config.interpolation_factor,
                projection_accuracy=self.config.projection_accuracy,
                supercell=self.config.supercell,
                max_distance=self.config.max_distance,
            )
            if fermi_surface3D.is_empty_mesh():
                fermi_surfaces.append(fermi_surface3D)
                continue

            self.property_name = property_name

            band_to_surface_indices = list(
                fermi_surface3D.band_isosurface_index_map.keys()
            )

            if self.property_name == "fermi_speed":
                fermi_surface3D.project_fermi_speed(
                    fermi_speed=ebs.fermi_speed[..., band_to_surface_indices, ispin]
                )
                if self.config.scalar_bar_config.get("title") is None:
                    self.config.scalar_bar_config["title"] = "Fermi Speed"
            elif self.property_name == "fermi_velocity":
                logger.debug(f"ebs.fermi_velocity shape: {ebs.fermi_velocity.shape}")

                fermi_surface3D.project_fermi_velocity(
                    fermi_velocity=ebs.fermi_velocity[
                        ..., band_to_surface_indices, ispin, :
                    ]
                )
                if self.config.scalar_bar_config.get("title") is None:
                    self.config.scalar_bar_config["title"] = "Fermi Velocity"
            elif self.property_name == "avg_inv_effective_mass":
                fermi_surface3D.project_avg_inv_effective_mass(
                    avg_inv_effective_mass=ebs.avg_inv_effective_mass[
                        ..., band_to_surface_indices, ispin
                    ]
                )
                if self.config.scalar_bar_config.get("title") is None:
                    self.config.scalar_bar_config["title"] = (
                        "Avg Inverse Effective Mass"
                    )
            if self.config.mode == "parametric":
                fermi_surface3D.project_atomic_projections(
                    self.spd[:, band_to_surface_indices, ispin]
                )
                if self.config.scalar_bar_config.get("title") is None:
                    self.config.scalar_bar_config["title"] = (
                        "Atomic Orbital Projections"
                    )
            if self.config.mode == "spin_texture":
                fermi_surface3D.project_spin_texture_atomic_projections(
                    self.spd_spin[:, band_to_surface_indices, :]
                )
                if self.config.scalar_bar_config.get("title") is None:
                    self.config.scalar_bar_config["title"] = "Spin Texture"

            if self.config.extended_zone_directions:
                fermi_surface3D.extend_surface(
                    extended_zone_directions=self.config.extended_zone_directions
                )
            fermi_surfaces.append(fermi_surface3D)

        self.fermi_surface = self._merge_fermi_surfaces(fermi_surfaces)
        logger.info(f"____ Retrieived Fermi Surface Data ____")
        return self.fermi_surface


class FermiVisualizer:

    def __init__(self, data_handler, config):

        self.data_handler = data_handler
        self.config = config
        self.plotter = pv.Plotter()

        self._setup_plotter()

    def add_scalar_bar(self, name, scalar_bar_config=None):
        if scalar_bar_config is None:
            scalar_bar_config = self.config.scalar_bar_config

        self.plotter.add_scalar_bar(**scalar_bar_config)

    def add_axes(self):
        self.plotter.add_axes(
            xlabel=self.config.x_axes_label,
            ylabel=self.config.y_axes_label,
            zlabel=self.config.z_axes_label,
            color=self.config.axes_label_color,
            line_width=self.config.axes_line_width,
            labels_off=False,
        )

    def add_brillouin_zone(self, fermi_surface):
        self.plotter.add_mesh(
            fermi_surface.brillouin_zone,
            style=self.config.brillouin_zone_style,
            line_width=self.config.brillouin_zone_line_width,
            color=self.config.brillouin_zone_color,
            opacity=self.config.brillouin_zone_opacity,
        )

    def add_surface(self, surface):
        logger.info(f"____Adding Surface to Plotter____")
        if surface.is_empty_mesh():
            logger.warning("No Fermi surface found. Skipping surface addition.")
            return None
        surface = self._setup_band_colors(surface)

        if self.config.spin_colors != (None, None):
            logger.debug(f"Adding surface with spin colors: {self.config.spin_colors}")
            spin_colors = []
            for spin_index in surface.point_data["spin_index"]:
                if spin_index == 0:
                    spin_colors.append(self.config.spin_colors[0])
                else:
                    spin_colors.append(self.config.spin_colors[1])
            surface.point_data["spin_colors"] = spin_colors
            self.plotter.add_mesh(
                surface,
                scalars="spin_colors",
                cmap=self.config.surface_cmap,
                clim=self.config.surface_clim,
                show_scalar_bar=False,
                opacity=self.config.surface_opacity,
            )

        elif self.config.surface_color:
            logger.debug(f"Adding surface with color: {self.config.surface_color}")
            self.plotter.add_mesh(
                surface,
                color=self.config.surface_color,
                opacity=self.config.surface_opacity,
            )
        else:
            logger.debug(
                f"Adding surface with scalars: {self.data_handler.scalars_name}"
            )
            logger.debug(f"surface: \n {surface}")
            logger.debug(f"surface.point_data: \n {surface.point_data}")
            logger.debug(f"surface.cell_data: \n {surface.cell_data.keys()}")
            # if self.config.surface_clim:
            # self._normalize_data(surface,scalars_name=self.data_handler.scalars_name)

            self.plotter.add_mesh(
                surface,
                scalars=self.data_handler.scalars_name,
                cmap=self.config.surface_cmap,
                clim=self.config.surface_clim,
                show_scalar_bar=False,
                opacity=self.config.surface_opacity,
                rgba=self.data_handler.use_rgba,
            )

    def add_texture(self, fermi_surface, scalars_name, vector_name):
        if vector_name not in fermi_surface.point_data:
            return None

        arrows = fermi_surface.glyph(
            orient=vector_name,
            scale=self.config.texture_scale,
            factor=self.config.texture_size,
        )
        if self.config.texture_color is None:
            self.plotter.add_mesh(
                arrows,
                scalars=scalars_name,
                cmap=self.config.texture_cmap,
                clim=self.config.texture_clim,
                show_scalar_bar=False,
                opacity=self.config.texture_opacity,
            )
        else:
            self.plotter.add_mesh(
                arrows,
                scalars=scalars_name,
                color=self.config.texture_color,
                show_scalar_bar=False,
                opacity=self.config.texture_opacity,
            )
        # else:
        #     arrows = None
        return arrows

    def add_isoslider(self, e_surfaces, energy_values):
        self.energy_values = energy_values
        self.e_surfaces = e_surfaces
        for i, surface in enumerate(self.e_surfaces):

            if self.config.spin_colors != (None, None):
                spin_colors = []
                for spin_index in surface.point_data["spin_index"]:
                    if spin_index == 0:
                        spin_colors.append(self.config.spin_colors[0])
                    else:
                        spin_colors.append(self.config.spin_colors[1])
                self.e_surfaces[i].point_data["spin_colors"] = spin_colors
            else:
                self.e_surfaces[i] = self._setup_band_colors(surface)

        self.plotter.add_slider_widget(
            self._custom_isoslider_callback,
            [np.amin(energy_values), np.amax(energy_values)],
            title=self.config.isoslider_title,
            style=self.config.isoslider_style,
            color=self.config.isoslider_color,
        )

        self.add_brillouin_zone(self.e_surfaces[0])
        self.add_axes()
        self.set_background_color()

    def add_isovalue_gif(self, e_surfaces, energy_values, save_gif):
        self.energy_values = energy_values
        self.e_surfaces = e_surfaces
        for i, surface in enumerate(self.e_surfaces):
            self.e_surfaces[i] = self._setup_band_colors(surface)

        self.add_brillouin_zone(self.e_surfaces[0])
        self.add_axes()
        self.set_background_color()

        self.plotter.off_screen = True
        self.plotter.open_gif(save_gif)
        # Initial Mesh
        surface = copy.deepcopy(e_surfaces[0])
        initial_e_value = energy_values[0]

        self.plotter.add_text(f"Energy Value : {initial_e_value:.4f} eV", color="black")
        if self.data_handler.vector_name:
            arrows = self.add_texture(
                surface,
                scalars_name=self.data_handler.scalars_name,
                vector_name=self.data_handler.vector_name,
            )
        self.add_surface(surface)
        if self.config.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)

        self.plotter.show(auto_close=False)

        # Run through each frame
        for e_surface, evalues in zip(e_surfaces, energy_values):

            surface.copy_from(e_surface)

            # Must set active scalars after calling copy_from
            surface.set_active_scalars(name=self.data_handler.scalars_name)

            text = f"Energy Value : {evalues:.4f} eV"
            self.plotter.text.SetText(2, text)

            if (
                self.data_handler.scalars_name == "spin_magnitude"
                or self.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
            ):
                e_arrows = e_surface.glyph(
                    orient=self.data_handler.vector_name,
                    scale=self.config.texture_scale,
                    factor=self.config.texture_size,
                )
                arrows.copy_from(e_arrows)
                arrows.set_active_scalars(name=self.data_handler.scalars_name)
                # arrows.set_active_vectors(name=options_dict['vector_name'])

            self.plotter.write_frame()

        # Run backward through each frame
        for e_surface, evalues in zip(e_surfaces[::-1], energy_values[::-1]):
            surface.copy_from(e_surface)

            # Must set active scalars after calling copy_from
            surface.set_active_scalars(name=self.data_handler.scalars_name)

            text = f"Energy Value : {evalues:.4f} eV"
            self.plotter.text.SetText(2, text)

            if (
                self.data_handler.scalars_name == "spin_magnitude"
                or self.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
            ):
                e_arrows = e_surface.glyph(
                    orient=self.data_handler.vector_name,
                    scale=self.config.texture_scale,
                    factor=self.config.texture_size,
                )

                arrows.copy_from(e_arrows)
                arrows.set_active_scalars(name=self.data_handler.scalars_name)
                # arrows.set_active_vectors(name=options_dict['vector_name'])

            self.plotter.write_frame()

        self.plotter.close()

    def add_slicer(
        self,
        surface,
        show=True,
        save_2d=None,
        save_2d_slice=None,
        slice_normal=(1, 0, 0),
        slice_origin=(0, 0, 0),
    ):
        if self.config.show_brillouin_zone:
            self.add_brillouin_zone(surface)
        if self.config.show_axes:
            self.add_axes()
        self.set_background_color()

        self.add_surface(surface)
        if self.config.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)

        if (
            self.data_handler.scalars_name == "spin_magnitude"
            or self.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
        ):
            self.add_texture(
                surface,
                scalars_name=self.data_handler.scalars_name,
                vector_name=self.data_handler.vector_name,
            )

        self._add_custom_mesh_slice(
            mesh=surface, normal=slice_normal, origin=slice_origin
        )

        if save_2d:
            self.savefig(save_2d)
            return None

        if show:
            self.show()

        if save_2d_slice:
            slice_2d = self.plotter.plane_sliced_meshes[0]
            self.plotter.close()
            point1 = slice_2d.points[0, :]
            point2 = slice_2d.points[1, :]
            normal_vec = np.cross(point1, point2)
            p = pv.Plotter()
            arrows = None
            if self.data_handler.vector_name:
                arrows = slice_2d.glyph(
                    orient=self.data_handler.vector_name,
                    scale=self.config.texture_scale,
                    factor=self.config.texture_size,
                )

            if arrows:
                if self.config.texture_color is not None:
                    p.add_mesh(
                        arrows,
                        color=self.config.texture_color,
                        show_scalar_bar=False,
                        name="arrows",
                    )
                else:
                    p.add_mesh(
                        arrows,
                        cmap=self.config.texture_cmap,
                        show_scalar_bar=False,
                        name="arrows",
                    )
            p.add_mesh(slice_2d, line_width=self.config.cross_section_slice_linewidth)
            p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice, interactive=False)

    def add_box_slicer(
        self,
        surface,
        show=True,
        save_2d=None,
        save_2d_slice=None,
        slice_normal=(1, 0, 0),
        slice_origin=(0, 0, 0),
    ):
        if self.config.show_brillouin_zone:
            self.add_brillouin_zone(surface)
        if self.config.show_axes:
            self.add_axes()
        self.set_background_color()

        self.add_surface(surface)
        if self.config.mode != "plain" and self.config.show_scalar_bar:
            self.add_scalar_bar(name=self.data_handler.scalars_name)
        if (
            self.data_handler.scalars_name == "spin_magnitude"
            or self.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
        ):
            self.add_texture(
                surface,
                scalars_name=self.data_handler.scalars_name,
                vector_name=self.data_handler.vector_name,
            )

        self._add_custom_box_slice_widget(
            mesh=surface,
            show_cross_section_area=self.config.cross_section_slice_show_area,
            normal=slice_normal,
            origin=slice_origin,
        )

        if save_2d:
            self.savefig(save_2d)
            return None

        if show:
            self.show()

        if save_2d_slice:

            slice_2d = self.plotter.plane_sliced_meshes[0]
            logger.debug(f"slice_2d point_data: \n {slice_2d.point_data}")
            self.plotter.close()
            point1 = slice_2d.points[0, :]
            point2 = slice_2d.points[1, :]
            point3 = slice_2d.points[3, :]
            normal_vec = np.cross(point1 - point3, point2 - point3)
            p = pv.Plotter()

            if self.data_handler.vector_name:
                arrows = slice_2d.glyph(
                    orient=self.data_handler.vector_name, scale=False, factor=0.1
                )
                if self.config.texture_color is not None:
                    p.add_mesh(
                        arrows,
                        color=self.config.texture_color,
                        show_scalar_bar=False,
                        name="arrows",
                    )
                else:
                    p.add_mesh(
                        arrows,
                        cmap=self.config.texture_cmap,
                        show_scalar_bar=False,
                        name="arrows",
                    )
            p.add_mesh(
                slice_2d,
                line_width=self.config.cross_section_slice_linewidth,
                cmap=self.config.surface_cmap,
            )
            if not self.config.show_scalar_bar:
                p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice, interactive=False)

    def set_background_color(self):
        self.plotter.set_background(self.config.background_color)

    def show(self, **kwargs):
        logger.info("Showing plot")

        cpos = self.plotter.show(return_cpos=True, **kwargs)
        self.plotter.camera_position = cpos
        user_message = (
            f"To save an image of where the camera is at time when the window closes,\n"
        )
        user_message += f"set the `save_2d` parameter and set `plotter_camera_pos` to the following: \n {cpos}"
        return user_message

    def savefig(self, filename, **kwargs):
        logger.info("Saving plot")

        if self.config.plotter_camera_pos:
            self.plotter.camera_position = self.config.plotter_camera_pos
        else:
            self.plotter.view_isometric()

        # Get the file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in [".pdf", ".eps", ".ps", ".tex", ".svg"]:
            self.plotter.save_graphic(filename)
        else:
            self.plotter.screenshot(filename)

    def save_gif(self, filename, save_gif_config=None):
        if save_gif_config is None:
            save_gif_config = self.config.save_gif_config

        path = self.plotter.generate_orbital_path(
            **save_gif_config["generate_orbital_path_kwargs"]
        )
        self.plotter.open_gif(filename, **save_gif_config["open_gif_kwargs"])
        self.plotter.orbit_on_path(
            path,
            write_frames=True,
            **save_gif_config["orbit_on_path_kwargs"],
        )

    def save_mp4(self, filename, save_mp4_config=None):
        if save_mp4_config is None:
            save_mp4_config = self.config.save_mp4_config

        path = self.plotter.generate_orbital_path(
            **save_mp4_config["generate_orbital_path_kwargs"]
        )
        self.plotter.open_movie(filename, **save_mp4_config["open_movie_kwargs"])
        self.plotter.orbit_on_path(
            path,
            write_frames=True,
            **save_mp4_config["orbit_on_path_kwargs"],
        )

    def save_mesh(self, filename, surface, save_mesh_config=None):
        if save_mesh_config is None:
            save_mesh_config = self.config.save_mesh_config

        pv.save_meshio(filename, surface, **save_mesh_config["save_meshio_kwargs"])

    def _setup_band_colors(self, fermi_surface):
        logger.info(f"____ Setting up Band Colors ____")

        band_colors = self.config.surface_bands_colors
        if self.config.surface_bands_colors == []:
            band_colors = self._generate_band_colors(fermi_surface)

        fermi_surface = self._apply_fermi_surface_band_colors(
            fermi_surface, band_colors
        )
        logger.info(f"____ Finished Setting up Band Colors ____")
        return fermi_surface

    def _generate_band_colors(self, fermi_surface):
        logger.info(f"____ Generating Band Colors ____")
        # Generate unique rgba values for the bands
        unique_band_index = np.unique(fermi_surface.point_data["band_index"])
        nsurface = len(unique_band_index)
        norm = mpcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap(self.config.surface_cmap)
        solid_color_surface = np.arange(nsurface) / nsurface
        band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(
            -1, 4
        )
        logger.debug(f"Band Colors: {band_colors}")
        logger.debug(f"Band Colors shape: {band_colors.shape}")
        logger.info(f"____ Finished Generating Band Colors ____")
        return band_colors

    def _apply_fermi_surface_band_colors(self, fermi_surface, band_colors):
        logger.info(f"____ Applying Band Colors ____")
        unique_band_index = np.unique(fermi_surface.point_data["band_index"])

        if len(band_colors) != len(unique_band_index):
            print("Number of bands : ", len(unique_band_index))
            raise "You need to list colors as there are bands that make up the surface."

        surface_band_colors = []
        for band_color in band_colors:
            if isinstance(band_color, str):
                surface_color = mpcolors.to_rgba_array(band_color, alpha=1)[0, :]
                surface_band_colors.append(surface_color)
            else:
                surface_color = band_color
                surface_band_colors.append(surface_color)

        band_colors = []
        band_surface_indices = fermi_surface.point_data["band_index"]
        for band_surface_index in band_surface_indices:
            band_color = surface_band_colors[band_surface_index]
            band_colors.append(band_color)

        fermi_surface.point_data["bands"] = band_colors
        logger.info(f"____ Finished Applying Band Colors ____")
        return fermi_surface

    def _setup_plotter(self):
        """Helper method set parameter values"""
        if self.config.mode == "plain":
            text = "plain"
            scalars = "bands"
            vector_name = None
            use_rgba = True

        elif self.config.mode == "parametric":
            text = "parametric"
            scalars = "scalars"
            vector_name = None
            use_rgba = False

        elif self.config.mode == "property_projection":

            use_rgba = False
            if self.config.property_name == "fermi_speed":
                scalars = "Fermi Speed"
                text = "Fermi Speed"
                vector_name = None
            elif self.config.property_name == "fermi_velocity":
                scalars = "Fermi Velocity Vector_magnitude"
                vector_name = "Fermi Velocity Vector"
                text = "Fermi Speed"
            elif self.config.property_name == "avg_inv_effective_mass":
                scalars = "Avg Inverse Effective Mass"
                text = "Avg Inverse Effective Mass"
                vector_name = None
            else:
                print("Please select a property")
        elif self.config.mode == "spin_texture":
            text = "Spin Texture"
            use_rgba = False
            scalars = "spin_magnitude"
            vector_name = "spin"

        self.data_handler.text = text
        self.data_handler.scalars_name = scalars
        self.data_handler.vector_name = vector_name
        self.data_handler.use_rgba = use_rgba

    def _normalize_data(self, surface, scalars_name):
        x = surface[scalars_name]
        vmin = self.config.surface_clim[0]
        vmax = self.config.surface_clim[1]
        x_max = x.max()
        x_min = x.min()
        x_norm = x_min + ((x - vmin) * (vmax - x_min)) / (x_max - x_min)
        surface[scalars_name] = x_norm
        return x_norm

    def _custom_isoslider_callback(self, value):
        res = float(value)
        closest_idx = find_nearest(self.energy_values, res)
        surface = self.e_surfaces[closest_idx]
        if self.config.surface_color:
            self.plotter.add_mesh(
                surface,
                name="iso_surface",
                color=self.config.surface_color,
                opacity=self.config.surface_opacity,
            )
        elif self.config.spin_colors != (None, None):
            self.plotter.add_mesh(
                surface,
                name="iso_surface",
                scalars="spin_colors",
                cmap=self.config.surface_cmap,
                clim=self.config.surface_clim,
                show_scalar_bar=False,
                opacity=self.config.surface_opacity,
            )
        else:
            if self.config.surface_clim:
                self._normalize_data(
                    surface, scalars_name=self.data_handler.scalars_name
                )
            self.plotter.add_mesh(
                surface,
                name="iso_surface",
                scalars=self.data_handler.scalars_name,
                cmap=self.config.surface_cmap,
                clim=self.config.surface_clim,
                show_scalar_bar=False,
                opacity=self.config.surface_opacity,
                rgba=self.data_handler.use_rgba,
            )

        if self.config.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)
        if (
            self.data_handler.scalars_name == "spin_magnitude"
            or self.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
        ):

            arrows = surface.glyph(
                orient=self.data_handler.vector_name,
                scale=self.config.texture_scale,
                factor=self.config.texture_size,
            )

            if self.config.texture_color is None:
                self.plotter.add_mesh(
                    arrows,
                    name="iso_texture",
                    scalars=self.data_handler.scalars_name,
                    cmap=self.config.texture_cmap,
                    show_scalar_bar=False,
                    opacity=self.config.texture_opacity,
                )
            else:
                self.plotter.add_mesh(
                    arrows,
                    name="iso_texture",
                    scalars=self.data_handler.scalars_name,
                    color=self.config.texture_color,
                    show_scalar_bar=False,
                    opacity=self.config.texture_opacity,
                )

        return None

    def _add_custom_mesh_slice(
        self,
        mesh,
        normal="x",
        generate_triangles=False,
        widget_color=None,
        assign_to_axis=None,
        tubing=False,
        origin_translation=True,
        origin=(0, 0, 0),
        outline_translation=False,
        implicit=True,
        normal_rotation=True,
        cmap="jet",
        arrow_color=None,
        **kwargs,
    ):

        line_width = self.config.cross_section_slice_linewidth
        #################################################################
        # Following code handles the plane clipping
        #################################################################

        name = kwargs.get("name", mesh.memory_address)

        if self.config.mode == "plain":
            rng = mesh.get_data_range(kwargs.get("bands", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            mesh.set_active_scalars(kwargs.get("bands", mesh.active_scalars_name))
        else:
            rng = mesh.get_data_range(kwargs.get("scalars", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            mesh.set_active_scalars(kwargs.get("scalars", mesh.active_scalars_name))

        self.plotter.add_mesh(
            mesh,
            name=name + "outline",
            opacity=0.0,
            scalars=self.data_handler.scalars_name,
            line_width=line_width,
            show_scalar_bar=False,
            rgba=self.data_handler.use_rgba,
        )

        alg = vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        self.plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pv.wrap(alg.GetOutput())
        self.plotter.plane_sliced_meshes.append(plane_sliced_mesh)

        def callback_plane(normal, origin):
            # create the plane for clipping

            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane)  # the cutter to use the plane we made
            alg.Update()  # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            # plotter.add_mesh(plane_sliced_mesh, name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])
            if self.data_handler.vector_name:
                arrows = plane_sliced_mesh.glyph(
                    orient=self.data_handler.vector_name, scale=False, factor=0.1
                )

                if arrow_color is not None:
                    self.plotter.add_mesh(
                        arrows, color=arrow_color, show_scalar_bar=False, name="arrows"
                    )
                else:
                    self.plotter.add_mesh(
                        arrows, cmap=cmap, show_scalar_bar=False, name="arrows"
                    )

        self.plotter.add_plane_widget(
            callback=callback_plane,
            bounds=mesh.bounds,
            factor=1.25,
            normal="x",
            color=widget_color,
            tubing=tubing,
            assign_to_axis=assign_to_axis,
            origin_translation=origin_translation,
            outline_translation=outline_translation,
            implicit=implicit,
            origin=origin,
            normal_rotation=normal_rotation,
        )

        actor = self.plotter.add_mesh(
            plane_sliced_mesh,
            show_scalar_bar=False,
            line_width=line_width,
            rgba=self.data_handler.use_rgba,
            **kwargs,
        )
        self.plotter.plane_widgets[0].SetNormal(normal)

        # Call the callback to update scene
        plane_origin = self.plotter.plane_widgets[0].GetOrigin()
        plane_normal = self.plotter.plane_widgets[0].GetNormal()
        callback_plane(normal=plane_normal, origin=plane_origin)
        return actor

    def _add_custom_box_slice_widget(
        self,
        mesh,
        show_cross_section_area: bool = False,
        line_width: float = 5.0,
        normal=(1, 0, 0),
        generate_triangles=False,
        widget_color=None,
        assign_to_axis=None,
        tubing=False,
        origin_translation=True,
        origin=(0, 0, 0),
        outline_translation=False,
        implicit=True,
        normal_rotation=True,
        cmap="jet",
        arrow_color=None,
        # box widget options
        invert=False,
        rotation_enabled=True,
        box_widget_color="black",
        box_outline_translation=True,
        merge_points=True,
        crinkle=False,
        interaction_event="end",
        **kwargs,
    ):

        #################################################################
        # Following code handles the box clipping
        #################################################################

        line_width = self.config.cross_section_slice_linewidth

        mesh = pv.PolyData(mesh)

        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True)
        )
        name = kwargs.get("name", mesh.memory_address)
        if self.config.mode == "plain":
            rng = mesh.get_data_range(kwargs.get("scalars", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            mesh.set_active_scalars(kwargs.get("scalars", mesh.active_scalars_name))
        else:
            rng = mesh.get_data_range(kwargs.get("bands", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            mesh.set_active_scalars(kwargs.get("bands", mesh.active_scalars_name))

        self.plotter.add_mesh(
            mesh,
            scalars=self.data_handler.scalars_name,
            show_scalar_bar=False,
            name=f"{name}-outline",
            opacity=0.0,
        )

        port = 1 if invert else 0

        clipper = vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            clipper.SetLocator(vtk.vtkNonMergingPointLocator())
        set_algorithm_input(clipper, algo)
        clipper.GenerateClippedOutputOn()

        if crinkle:
            crinkler = crinkle_algorithm(clipper.GetOutputPort(port), algo)
            box_clipped_mesh = _get_output(crinkler)
        else:
            box_clipped_mesh = _get_output(clipper, oport=port)

        self.plotter.box_clipped_meshes.append(box_clipped_mesh)

        def callback_box(planes):
            bounds = []
            for i in range(planes.GetNumberOfPlanes()):
                plane = planes.GetPlane(i)
                bounds.append(plane.GetNormal())
                bounds.append(plane.GetOrigin())

            clipper.SetBoxClip(*bounds)
            clipper.Update()
            if crinkle:
                clipped = pv.wrap(crinkler.GetOutputDataObject(0))
            else:
                clipped = _get_output(clipper, oport=port)
            box_clipped_mesh.shallow_copy(clipped)

            # Update plane widget after updating box widget
            plane_origin = self.plotter.plane_widgets[0].GetOrigin()
            plane_normal = self.plotter.plane_widgets[0].GetNormal()
            callback_plane(normal=plane_normal, origin=plane_origin)

        #################################################################
        # Following code handles the plane clipping
        #################################################################

        clipped_box_mesh = self.plotter.box_clipped_meshes[0]

        name = kwargs.get("name", clipped_box_mesh.memory_address)

        if self.config.mode == "plain":
            rng = clipped_box_mesh.get_data_range(kwargs.get("scalars", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            clipped_box_mesh.set_active_scalars(
                kwargs.get("scalars", clipped_box_mesh.active_scalars_name)
            )
        else:
            rng = clipped_box_mesh.get_data_range(kwargs.get("bands", None))
            kwargs.setdefault("clim", kwargs.pop("rng", rng))
            clipped_box_mesh.set_active_scalars(
                kwargs.get("bands", clipped_box_mesh.active_scalars_name)
            )

        # self.plotter.add_mesh(clipped_box_mesh,
        #                       name=name+"outline",
        #                       opacity=0.0,
        #                       line_width=0.0,
        #                       show_scalar_bar=False,
        #                       rgba=self.data_handler.use_rgba)

        alg = vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(
            clipped_box_mesh
        )  # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        self.plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pv.wrap(alg.GetOutput())
        self.plotter.plane_sliced_meshes.append(plane_sliced_mesh)

        if show_cross_section_area:
            user_slice = self.plotter.plane_sliced_meshes[0]
            surface = user_slice.delaunay_2d()
            self.plotter.add_text(
                f"Cross sectional area : {surface.area:.4f}" + " Ang^-2", color="black"
            )

        def callback_plane(normal, origin):
            # create the plane for clipping

            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane)  # the cutter to use the plane we made
            alg.Update()  # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            # plotter.add_mesh(plane_sliced_mesh, name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])
            if self.data_handler.vector_name:
                arrows = plane_sliced_mesh.glyph(
                    orient=self.data_handler.vector_name, scale=False, factor=0.1
                )

                if arrow_color is not None:
                    self.plotter.add_mesh(
                        arrows, color=arrow_color, show_scalar_bar=False, name="arrows"
                    )
                else:
                    self.plotter.add_mesh(
                        arrows, cmap=cmap, show_scalar_bar=False, name="arrows"
                    )

            if show_cross_section_area:
                user_slice = self.plotter.plane_sliced_meshes[0]
                surface = user_slice.delaunay_2d()
                text = f"Cross sectional area : {surface.area:.4f}" + " Ang^-2"

                self.plotter.text.SetText(2, text)

        self.plotter.add_plane_widget(
            callback=callback_plane,
            bounds=mesh.bounds,
            factor=1.25,
            normal=normal,
            color=widget_color,
            tubing=tubing,
            assign_to_axis=assign_to_axis,
            origin_translation=origin_translation,
            outline_translation=outline_translation,
            implicit=implicit,
            origin=origin,
            normal_rotation=normal_rotation,
        )

        self.plotter.add_box_widget(
            callback=callback_box,
            bounds=mesh.bounds,
            factor=1.25,
            rotation_enabled=rotation_enabled,
            use_planes=True,
            color=box_widget_color,
            outline_translation=box_outline_translation,
            interaction_event=interaction_event,
        )

        actor = self.plotter.add_mesh(
            plane_sliced_mesh,
            show_scalar_bar=False,
            line_width=line_width,
            rgba=self.data_handler.use_rgba,
            **kwargs,
        )
        self.plotter.plane_widgets[0].SetNormal(normal)

        # # Call the callback to update scene
        plane_origin = self.plotter.plane_widgets[0].GetOrigin()
        plane_normal = self.plotter.plane_widgets[0].GetNormal()
        callback_plane(normal=plane_normal, origin=plane_origin)
        return actor


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
