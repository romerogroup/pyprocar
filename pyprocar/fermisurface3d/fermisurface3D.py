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

import math
HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg
class FermiSurfaceBand3D(Isosurface):

    def __init__(
        self,
        kpoints=None,
        band=None,
        spd=None,
        spd_spin=None,
        fermi_velocity_vector = False,
        fermi_velocity =False,
        effective_mass = False,
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
        sym =False
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
        fermi_velocity_vector : bool, optional (default False)
            Boolean value to calculate fermi velocity vectors on the band surface
            e.g. ``fermi_velocity_vector=True``
        fermi_velocity : bool, optional (default False)
            Boolean value to calculate magnitude of the fermi velocity on the band surface
            e.g. ``fermi_velocity=True``
        effective_mass : bool, optional (default False)
            Boolean value to calculate the harmonic mean of the effective mass on the band surface
            e.g. ``effective_mass=True``
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
        self.sym = sym
        
        self.fermi_velocity_vector = fermi_velocity_vector
        self.fermi_velocity = fermi_velocity
        self.effective_mass = effective_mass
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)


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
            self.project_color(cmap, vmin, vmax, scalars = self.spd)
        if self.spd_spin is not None and self.verts is not None:
            self.create_spin_texture( vectors = self.spd_spin)
            
        if self.fermi_velocity_vector is not None and self.verts is not None:
            self.calculate_first_and_second_derivative_energy()
            self.create_vector_texture( vectors = self.group_velocity_vector)
            
        if self.fermi_velocity == True and self.verts is not None:
            self.calculate_first_and_second_derivative_energy()
            self.project_color(cmap, vmin, vmax, scalars = self.group_velocity_magnitude)
        if self.effective_mass == True and self.verts is not None:
            self.calculate_first_and_second_derivative_energy()
            self.project_color(cmap, vmin, vmax, scalars = self.effective_mass_list)

        if self.sym == True:
            self.ibz2fbz()

    def create_vector_texture(self,vectors):


        XYZ_extended = self.XYZ.copy()
        vectors_extended_X = vectors[0].copy()
        vectors_extended_Y = vectors[1].copy()
        vectors_extended_Z = vectors[2].copy()

        for ix in range(3):
            for iy in range(self.supercell[ix]):
                temp = self.XYZ.copy()
                temp[:, ix] += 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )
                temp = self.XYZ.copy()
                temp[:, ix] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )

        if np.any(self.XYZ >= 0.5): # @logan : I think this should be before the above loop with another else statment
            for iy in range(self.supercell[ix]):
                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 1] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )
                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )
                temp = self.XYZ.copy()
                temp[:, 1] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )

                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 1] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                vectors_extended_X = np.append(
                    vectors_extended_X, vectors[0], axis=0
                )
                vectors_extended_Y = np.append(
                    vectors_extended_Y, vectors[1], axis=0
                )
                vectors_extended_Z = np.append(
                    vectors_extended_Z, vectors[2], axis=0
                )

        # XYZ_extended = self.XYZ.copy()
        # scalars_extended = self.spd.copy()

        XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
        # XYZ_transformed = XYZ_extended

        if self.projection_accuracy.lower()[0] == "n":

            vectors_X = interpolate.griddata(
                XYZ_transformed, vectors_extended_X, self.verts, method="nearest"
            )
            vectors_Y = interpolate.griddata(
                XYZ_transformed, vectors_extended_Y, self.verts, method="nearest"
            )
            vectors_Z = interpolate.griddata(
                XYZ_transformed, vectors_extended_Z, self.verts, method="nearest"
            )

        elif self.projection_accuracy.lower()[0] == "h":

            vectors_X = interpolate.griddata(
                XYZ_transformed, vectors_extended_X, self.verts, method="linear"
            )
            vectors_Y = interpolate.griddata(
                XYZ_transformed, vectors_extended_Y, self.verts, method="linear"
            )
            vectors_Z = interpolate.griddata(
                XYZ_transformed, vectors_extended_Z, self.verts, method="linear"
            )

        self.set_vectors(vectors_X, vectors_Y, vectors_Z)
            
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

            if np.any(self.XYZ >= 0.5): # @logan : I think this should be before the above loop with another else statment
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
   
    def project_color(self, cmap, vmin, vmax, scalars ):
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

        XYZ_extended = self.XYZ.copy()
        scalars_extended =  scalars.copy()


        for ix in range(3):
            for iy in range(self.supercell[ix]):
                temp = self.XYZ.copy()
                temp[:, ix] += 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)
                temp = self.XYZ.copy()
                temp[:, ix] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)
        if np.any(self.XYZ >= 0.5): # @logan same here
            for iy in range(self.supercell[ix]):
                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 1] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)
                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)
                temp = self.XYZ.copy()
                temp[:, 1] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)

                temp = self.XYZ.copy()
                temp[:, 0] -= 1 * (iy + 1)
                temp[:, 1] -= 1 * (iy + 1)
                temp[:, 2] -= 1 * (iy + 1)
                XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                scalars_extended = np.append(scalars_extended,  scalars, axis=0)

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
              
    def calculate_first_and_second_derivative_energy(self):
        def get_energy(kp_reduced):
            return kp_reduced_to_energy[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def get_cartesian_kp(kp_reduced):
            return kp_reduced_to_kp_cart[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def energy_meshgrid_mapping(mesh_list):
            import copy 
            mesh_list_cart = copy.deepcopy(mesh_list)
            F = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
        
                        mesh_list_cart[0][i,j,k] = get_cartesian_kp([kx,ky,kz])[0]
                        mesh_list_cart[1][i,j,k] = get_cartesian_kp([kx,ky,kz])[1]
                        mesh_list_cart[2][i,j,k] = get_cartesian_kp([kx,ky,kz])[2]
                        F[i,j,k] = get_energy([kx,ky,kz])
        
            return F
        def fermi_velocity_meshgrid_mapping(mesh_list):
            X = np.zeros(shape = mesh_list[0].shape)
            Y = np.zeros(shape = mesh_list[0].shape)
            Z = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
                        
                        X[i,j,k] = get_gradient([kx,ky,kz])[0]
                        Y[i,j,k] = get_gradient([kx,ky,kz])[1]
                        Z[i,j,k] = get_gradient([kx,ky,kz])[2]
        
            return X,Y,Z
        
        kpoints_cart = np.matmul(self.XYZ,  self.reciprocal_lattice)
        
        mesh_list = np.meshgrid(np.unique(self.XYZ[:,0]),
                                np.unique(self.XYZ[:,1]),
                                np.unique(self.XYZ[:,2]), indexing = 'ij')
        
        
        kp_reduced_to_energy = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,self.band)}
        kp_reduced_to_kp_cart = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,kpoints_cart)}
        kp_reduced_to_mesh_index = {f'({kp[0]},{kp[1]},{kp[2]})': np.argwhere((mesh_list[0]==kp[0]) & 
                                                                              (mesh_list[1]==kp[1]) & 
                                                                              (mesh_list[2]==kp[2]) )[0] for kp in self.XYZ}
        
        
        energy_meshgrid = energy_meshgrid_mapping(mesh_list) 

        change_in_energy = np.gradient(energy_meshgrid)

        grad_x_list = [change_in_energy[0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                  kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                  kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                  ]
                  for kp in self.XYZ]
        grad_y_list = [change_in_energy[1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                              ]
                  for kp in self.XYZ]
        grad_z_list = [change_in_energy[2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                              ]
                  for kp in self.XYZ]
        
        gradient_list = np.array([grad_x_list,grad_y_list,grad_z_list]).T

        lattice = np.linalg.inv(self.reciprocal_lattice.T).T

      
        gradient_list = gradient_list/(2*math.pi)
        gradient_list = np.multiply(gradient_list, np.array([np.linalg.norm(lattice[:,0])*METER_ANGSTROM * len(np.unique(self.XYZ[:,0])),
                                                             np.linalg.norm(lattice[:,1])*METER_ANGSTROM * len(np.unique(self.XYZ[:,1])),
                                                             np.linalg.norm(lattice[:,2])*METER_ANGSTROM * len(np.unique(self.XYZ[:,2]))])
                            )
        
        gradient_list_cart = np.array([np.matmul(lattice, gradient) for gradient in gradient_list])
        
 
        
        # kp_reduced_to_grad_x = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_x_list)}
        # kp_reduced_to_grad_y = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_y_list)}
        # kp_reduced_to_grad_z = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_z_list)}
        
        self.group_velocity_x = gradient_list_cart[:,0]/HBAR_EV
        self.group_velocity_y = gradient_list_cart[:,1]/HBAR_EV
        self.group_velocity_z = gradient_list_cart[:,2]/HBAR_EV
        
        self.group_velocity_vector = [self.group_velocity_x,self.group_velocity_y,self.group_velocity_z]
        
        self.group_velocity_magnitude = (self.group_velocity_x**2 + 
                                         self.group_velocity_y**2 + 
                                         self.group_velocity_z**2)**0.5
        
        if self.effective_mass == True:
            kp_reduced_to_gradient = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,gradient_list_cart)}
            def get_gradient(kp_reduced):
                return kp_reduced_to_gradient[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
            
            gradient_mesh = list(fermi_velocity_meshgrid_mapping(mesh_list))
            change_in_energy_gradient = list(map(np.gradient,gradient_mesh))
            
            
            
            grad_xx_list = [change_in_energy_gradient[0][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_xy_list = [change_in_energy_gradient[0][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_xz_list = [change_in_energy_gradient[0][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            
            grad_yx_list = [change_in_energy_gradient[1][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_yy_list = [change_in_energy_gradient[1][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_yz_list = [change_in_energy_gradient[1][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            
            grad_zx_list = [change_in_energy_gradient[2][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_zy_list = [change_in_energy_gradient[2][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            grad_zz_list = [change_in_energy_gradient[2][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                            kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                            ]
                            for kp in self.XYZ]
            
            energy_second_derivative_list = np.array([[grad_xx_list,grad_xy_list,grad_xz_list],
                                             [grad_yx_list,grad_yy_list,grad_yz_list],
                                             [grad_zx_list,grad_zy_list,grad_zz_list],
                                             ])
            
            energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=0,axis2=2)
            energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=1,axis2=2)
            energy_second_derivative_list = energy_second_derivative_list/(2*math.pi)
            energy_second_derivative_list = np.multiply(energy_second_derivative_list, np.array([np.linalg.norm(lattice[:,0])*METER_ANGSTROM * len(np.unique(self.XYZ[:,0])),
                                                                                                           np.linalg.norm(lattice[:,1])*METER_ANGSTROM * len(np.unique(self.XYZ[:,1])),
                                                                                                           np.linalg.norm(lattice[:,2])*METER_ANGSTROM * len(np.unique(self.XYZ[:,2]))])
                                                        )
            self.energy_second_derivative_list_cart = energy_second_derivative_list_cart = np.array([ [ np.matmul(lattice,gradient) for gradient in gradient_tensor ] for gradient_tensor in energy_second_derivative_list ])
            
            self.inverse_effective_mass_tensor_list = self.energy_second_derivative_list_cart * EV_TO_J/ HBAR_J**2
            self.effective_mass_tensor_list = effective_mass_tensor_list = np.array([ np.linalg.inv(inverse_effective_mass_tensor) for inverse_effective_mass_tensor in self.inverse_effective_mass_tensor_list])
            self.effective_mass_list = np.array([ (3*(1/effective_mass_tensor_list[ikp,0,0] + 
                                                      1/effective_mass_tensor_list[ikp,1,1] +
                                                      1/effective_mass_tensor_list[ikp,2,2])**-1)/FREE_ELECTRON_MASS for ikp in range(len(self.XYZ))])
            
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
        fermi_velocity = False,
        fermi_velocity_vector = False,
        effective_mass = False,
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

        bands : (n,i) float
            An array of energies cooresponding to the
            kpoints and bands.

        spd :
            numpy array containing the information about ptojection of atoms,
            orbitals and spin on each band (check procarparser)
            
        fermi_velocity_vector : bool, optional (default False)
            Boolean value to calculate fermi velocity vectors on the fermi surface
            e.g. ``fermi_velocity_vector=True``
        fermi_velocity : bool, optional (default False)
            Boolean value to calculate magnitude of the fermi velocity on the fermi surface
            e.g. ``fermi_velocity=True``
        effective_mass : bool, optional (default False)
            Boolean value to calculate the harmonic mean of the effective mass on the fermi surface
            e.g. ``effective_mass=True``
            
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
        
        self.fermi_velocity = fermi_velocity
        self.fermi_velocity_vector = fermi_velocity_vector
        self.effective_mass = effective_mass
        
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
                fermi_velocity = self.fermi_velocity,
                fermi_velocity_vector=self.fermi_velocity_vector,
                effective_mass = self.effective_mass,
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

    # def ibz2fbz(self):
    #     """
    #     Converts the irreducible Brilluoin zone to the full Brillouin zone.

    #     Parameters:
    #     """
    #     klist = []
    #     bandlist = []
    #     spdlist = []

    #     for i, _ in enumerate(self.rotations):
    #         # for each point
    #         for j, _ in enumerate(self.kpoints):
    #             # apply symmetry operation to kpoint
    #             sympoint_vector = np.dot(self.rotations[i], self.kpoints[j])
    #             # apply boundary conditions
    #             # bound_ops = -1.0*(sympoint_vector > 0.5) + 1.0*(sympoint_vector < -0.5)
    #             # sympoint_vector += bound_ops

    #             sympoint = sympoint_vector.tolist()

    #             if sympoint not in klist:
    #                 klist.append(sympoint)

    #                 if self.band is not None:
    #                     band = self.band[j].tolist()
    #                     bandlist.append(band)
    #                 if self.spd is not None:
    #                     spd = self.spd[j].tolist()
    #                     spdlist.append(spd)

    #     self.kpoints = np.array(klist)
    #     self.band = np.array(bandlist)
    #     self.spd = np.array(spdlist)
