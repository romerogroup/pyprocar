"""
Created on Fri March 31 2020
@author: Pedram Tavadze

"""
import numpy as np
import itertools
import scipy.interpolate as interpolate
from ..core import Isosurface
from .brillouin_zone import BrillouinZone
from matplotlib import colors as mpcolors
from matplotlib import cm


class FermiSurfaceBand3D(Isosurface):
    def __init__(
        self,
        kpoints=None,
        band=None,
        spd=None,
        spd_spin=None,
        fermi=None,
        reciprocal_lattice=None,
        interpolation_factor=1,
        spin_texture=False,
        color=None,
        projection_accuracy="Normal",
        cmap="viridis",
        vmin=0,
        vmax=1,
        supercell=[1, 1, 1],
    ):
        """

        Parameters
        ----------
        kpoints : (n,3) float
            A list of kpoints used in the DFT calculation, this list
            has to be (n,3), n being number of kpoints and 3 being the
            3 different cartesian coordinates.

        band : (n,) float
            A list of energies of ith band cooresponding to the
            kpoints.
        spd :
            numpy array containing the information about ptojection of atoms,
            orbitals and spin on each band (check procarparser)
        fermi : float
            Value of the fermi energy or any energy that one wants to
            find the isosurface with.
        reciprocal_lattice : (3,3) float
            Reciprocal lattice of the structure.
        interpolation_factor : int
            The default is 1. number of kpoints in every direction
            will increase by this factor.
        color : TYPE, optional
            DESCRIPTION. The default is None.
        projection_accuracy : TYPE, optional
            DESCRIPTION. The default is 'Normal'.
        cmap : str
            The default is 'viridis'. Color map used in projecting the
            colors on the surface
        vmin : TYPE, float
            DESCRIPTION. The default is 0.
        vmax : TYPE, float
            DESCRIPTION. The default is 1.
        """

        self.kpoints = kpoints
        self.band = band
        self.spd = spd
        self.reciprocal_lattice = reciprocal_lattice
        self.supercell = np.array(supercell)
        self.fermi = fermi
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.spin_texture = spin_texture
        self.spd_spin = spd_spin

        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        # self.brillouin_zone = None

        if np.any(self.kpoints >= 0.5):
            Isosurface.__init__(
                self,
                XYZ=self.kpoints,
                V=self.band,
                isovalue=self.fermi,
                algorithm="lewiner",
                interpolation_factor=interpolation_factor,
                padding=self.supercell * 2,
                transform_matrix=self.reciprocal_lattice,
                boundaries=self.brillouin_zone,
            )
        else:
            Isosurface.__init__(
                self,
                XYZ=self.kpoints,
                V=self.band,
                isovalue=self.fermi,
                algorithm="lewiner",
                interpolation_factor=interpolation_factor,
                padding=self.supercell,
                transform_matrix=self.reciprocal_lattice,
                boundaries=self.brillouin_zone,
            )
        if self.spd is not None and self.verts is not None:
            self.project_color(cmap, vmin, vmax)
        if self.spd_spin is not None and self.verts is not None:
            self.create_spin_texture()

    def create_spin_texture(self):

        if self.spd_spin is not None:
            XYZ_extended = self.XYZ.copy()
            vectors_extended_X = self.spd_spin[0].copy()
            vectors_extended_Y = self.spd_spin[1].copy()
            vectors_extended_Z = self.spd_spin[2].copy()

            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

            if np.any(self.XYZ >= 0.5):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()

            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
            # XYZ_transformed = XYZ_extended

            if self.projection_accuracy.lower()[0] == "n":

                spin_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, self.verts, method="nearest"
                )
                spin_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, self.verts, method="nearest"
                )
                spin_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, self.verts, method="nearest"
                )

            elif self.projection_accuracy.lower()[0] == "h":

                spin_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, self.verts, method="linear"
                )
                spin_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, self.verts, method="linear"
                )
                spin_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, self.verts, method="linear"
                )

            self.set_vectors(spin_X, spin_Y, spin_Z)

    def project_color(self, cmap, vmin, vmax):
        """
        Projects the scalars to the surface.
        Parameters
        ----------
        cmap : TYPE string
            DESCRIPTION. Colormaps for the projection.
        vmin : TYPE
            DESCRIPTION.
        vmax : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        """
        if self.spd is not None:
            XYZ_extended = self.XYZ.copy()
            scalars_extended = self.spd.copy()

            # translations = itertools.product([-1,1,0],repeat = 3)

            # for trans in translations:
            #     for ix in range(len(trans)):
            #         for iy in range( self.supercell[ix]):
            #             temp = self.XYZ.copy()
            #             temp[:, 0] += trans[0] * (iy + 1)
            #             temp[:, 1] += trans[1] * (iy + 1)
            #             temp[:, 2] += trans[2] * (iy + 1)
            #             XYZ_extended = np.append(XYZ_extended, temp, axis=0)
            #             scalars_extended = np.append(scalars_extended,
            #                                           self.spd,
            #                                           axis=0)

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()

            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)
            if np.any(self.XYZ >= 0.5):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)
                    temp = self.XYZ.copy()
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)

                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended, self.spd, axis=0)

            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)

            # XYZ_transformed = XYZ_extended

            if self.projection_accuracy.lower()[0] == "n":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, self.centers, method="nearest"
                )
            elif self.projection_accuracy.lower()[0] == "h":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, self.centers, method="linear"
                )

            self.set_scalars(colors)
            self.set_color_with_cmap(cmap, vmin, vmax)

    def _get_brilloin_zone(self, supercell):
        return BrillouinZone(self.reciprocal_lattice, supercell)


class FermiSurface3D:
    def __init__(
        self,
        kpoints=None,
        bands=None,
        band_numbers=None,
        spd=None,
        spd_spin=None,
        fermi=None,
        fermi_shift=None,
        reciprocal_lattice=None,
        extended_zone_directions=None,
        interpolation_factor=1,
        spin_texture=False,
        colors=None,
        projection_accuracy="Normal",
        curvature_type = 'mean',
        cmap="viridis",
        vmin=0,
        vmax=1,
        supercell=[1, 1, 1],
    ):
        """


        Parameters
        ----------
        kpoints : (n,3) float
            A list of kpoints used in the DFT calculation, this list
            has to be (n,3), n being number of kpoints and 3 being the
            3 different cartesian coordinates.

        band : (n,) float
            A list of energies of ith band cooresponding to the
            kpoints.

        spd :
            numpy array containing the information about ptojection of atoms,
            orbitals and spin on each band (check procarparser)

        fermi : float
            Value of the fermi energy or any energy that one wants to
            find the isosurface with.

        reciprocal_lattice : (3,3) float
            Reciprocal lattice of the structure.

        interpolation_factor : int
            The default is 1. number of kpoints in every direction
            will increase by this factor.

        color : TYPE, optional
            DESCRIPTION. The default is None.

        projection_accuracy : TYPE, optional
            DESCRIPTION. The default is 'Normal'.

        cmap : str
            The default is 'viridis'. Color map used in projecting the
            colors on the surface

        vmin : TYPE, float
            DESCRIPTION. The default is 0.

        vmax : TYPE, float
            DESCRIPTION. The default is 1.

        """

        self.kpoints = kpoints
        self.bands = bands
        self.band_numbers = band_numbers
        self.spd = spd
        self.reciprocal_lattice = reciprocal_lattice
        self.supercell = np.array(supercell)
        self.fermi = fermi
        self.fermi_shift = fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.spin_texture = spin_texture
        self.spd_spin = spd_spin
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)

        self.colors = colors

        self.band_surfaces_obj = []
        self.band_surfaces = []
        self.band_surfaces_area = []
        self.band_surfaces_curvature = [[None]]

        self.fermi_surface = None
        self.fermi_surface_area = None

        self.fermi_surface_curvature = None

        counter = 0
        for iband in self.band_numbers:
            print("Trying to extract isosurface for band %d" % iband)

            surface = FermiSurfaceBand3D(
                kpoints=self.kpoints,
                band=self.bands[:, iband],
                spd=self.spd[counter],
                spd_spin=self.spd_spin[counter],
                fermi=self.fermi + self.fermi_shift,
                reciprocal_lattice=self.reciprocal_lattice,
                interpolation_factor=self.interpolation_factor,
                projection_accuracy=self.projection_accuracy,
                supercell=self.supercell,
            )

            # if surface.verts is not None:
            #     self.band_surfaces.append(surface)
            if surface.verts is not None:
                self.band_surfaces_obj.append(surface)
                self.band_surfaces_area.append(surface.pyvista_obj.area)
                self.band_surfaces.append(surface.pyvista_obj)
                self.band_surfaces_curvature.append(surface.pyvista_obj.curvature(curv_type=curvature_type))
            counter += 1

        nsurface = len(self.band_surfaces)
        norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)

        cmap = cm.get_cmap(cmap)
        scalars = np.arange(nsurface + 1) / nsurface

        if self.colors is None:

            self.colors = np.array([cmap(norm(x)) for x in (scalars)]).reshape(-1, 4)

        extended_surfaces = []
        extended_colors = []
        if extended_zone_directions is not None:
            for isurface in range(len(self.band_surfaces)):
                # extended_surfaces.append(self.band_surfaces[isurface].pyvista_obj)
                extended_surfaces.append(self.band_surfaces[isurface])
                extended_colors.append(self.colors[isurface])
            for direction in extended_zone_directions:
                for isurface in range(len(self.band_surfaces)):
                    # surface = self.band_surfaces[isurface].pyvista_obj.copy()
                    surface = self.band_surfaces[isurface].copy()
                    surface.translate(np.dot(direction, reciprocal_lattice))
                    extended_surfaces.append(surface)
                    extended_colors.append(self.colors[isurface])
            extended_colors.append(self.colors[-1])
            self.band_surfaces = extended_surfaces
            nsurface = len(extended_surfaces)
            self.colors = extended_colors

        self.fermi_surface = self.band_surfaces[0]
        for isurface in range(1, nsurface):
            self.fermi_surface = self.fermi_surface + self.band_surfaces[isurface]

        self.fermi_surface_area = self.fermi_surface.area
        self.fermi_surface_curvature = self.fermi_surface.curvature(curv_type=curvature_type)

    def _get_brilloin_zone(self, supercell):
        return BrillouinZone(self.reciprocal_lattice, supercell)
