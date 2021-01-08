"""
Created on Fri March 31 2020
@author: Pedram Tavadze

"""
import numpy as np
import scipy.interpolate as interpolate
from ..core import Isosurface
from .brillouin_zone import BrillouinZone


class FermiSurface3D(Isosurface):
    def __init__(self,
                 kpoints=None,
                 band=None,
                 spd=None,
                 spd_spin=None,
                 fermi=None,
                 reciprocal_lattice=None,
                 interpolation_factor=1,
                 spin_texture=False,
                 color=None,
                 projection_accuracy='Normal',
                 cmap='viridis',
                 vmin=0,
                 vmax=1,
                 supercell=[1, 1, 1],
                 file = None):
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
        self.file = file
        
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        #self.brillouin_zone = None
      
        if self.file == "bxsf" or self.file =='qe' or self.file == 'lobster':
            Isosurface.__init__(self,
                                XYZ=self.kpoints,
                                V=self.band,
                                isovalue=self.fermi,
                                algorithm='lewiner',
                                interpolation_factor=interpolation_factor,
                                padding=self.supercell*2,
                                transform_matrix=self.reciprocal_lattice,
                                boundaries=self.brillouin_zone,
                                file = self.file)
        else:
            Isosurface.__init__(self,
                                XYZ=self.kpoints,
                                V=self.band,
                                isovalue=self.fermi,
                                algorithm='lewiner',
                                interpolation_factor=interpolation_factor,
                                padding=self.supercell,
                                transform_matrix=self.reciprocal_lattice,
                                boundaries=self.brillouin_zone)
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
                    vectors_extended_X = np.append(vectors_extended_X,
                                                   self.spd_spin[0],
                                                   axis=0)
                    vectors_extended_Y = np.append(vectors_extended_Y,
                                                   self.spd_spin[1],
                                                   axis=0)
                    vectors_extended_Z = np.append(vectors_extended_Z,
                                                   self.spd_spin[2],
                                                   axis=0)
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(vectors_extended_X,
                                                   self.spd_spin[0],
                                                   axis=0)
                    vectors_extended_Y = np.append(vectors_extended_Y,
                                                   self.spd_spin[1],
                                                   axis=0)
                    vectors_extended_Z = np.append(vectors_extended_Z,
                                                   self.spd_spin[2],
                                                   axis=0)

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()

            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
            # XYZ_transformed = XYZ_extended

            if self.projection_accuracy.lower()[0] == 'n':

                spin_X = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_X,
                                              self.verts,
                                              method="nearest")
                spin_Y = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_Y,
                                              self.verts,
                                              method="nearest")
                spin_Z = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_Z,
                                              self.verts,
                                              method="nearest")

            elif self.projection_accuracy.lower()[0] == 'h':

                spin_X = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_X,
                                              self.verts,
                                              method="linear")
                spin_Y = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_Y,
                                              self.verts,
                                              method="linear")
                spin_Z = interpolate.griddata(XYZ_transformed,
                                              vectors_extended_Z,
                                              self.verts,
                                              method="linear")

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

            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,
                                                 self.spd,
                                                 axis=0)
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,
                                                 self.spd,
                                                 axis=0)

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()

            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
            #XYZ_transformed = XYZ_extended

            if self.projection_accuracy.lower()[0] == 'n':
                colors = interpolate.griddata(XYZ_transformed,
                                              scalars_extended,
                                              self.centers,
                                              method="nearest")
            elif self.projection_accuracy.lower()[0] == 'h':
                colors = interpolate.griddata(XYZ_transformed,
                                              scalars_extended,
                                              self.centers,
                                              method="linear")

            self.set_scalars(colors)
            self.set_color_with_cmap(cmap, vmin, vmax)

    def _get_brilloin_zone(self, supercell):
        return BrillouinZone(self.reciprocal_lattice, supercell)
