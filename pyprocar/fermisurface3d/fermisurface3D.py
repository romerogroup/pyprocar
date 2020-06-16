"""
Created on Fri March 31 2020
@author: Pedram Tavadze

"""
import numpy as np
import scipy.interpolate as interpolate
from .isosurface import Isosurface
from .brillouin_zone import BrillouinZone

class FermiSurface3D(Isosurface):
    def __init__(self,
                 kpoints=None,
                 band=None,
                 spd=None,
                 fermi=None,
                 reciprocal_lattice=None,
                 interpolation_factor=1,
                 color=None,
                 projection_accuracy='Normal',
                 cmap='viridis',
                 vmin=0,
                 vmax=1,
                 ):
        """
        
        Parameters
        ----------
        kpoints : (n,3) float
            A list of kpoints used in the DFT calculation, this list has to be
            (n,3), n being number of kpoints and 3 being the 3 different 
            cartesian coordinates
        band : (n,) float
            A list of energies of ith band cooresponding to the kpoints
        spd : 
            numpy array containing the information about ptojection of atoms,
            orbitals and spin on each band (check procarparser)
        fermi : float
            value of the fermi energy or any energy that one wants to find the 
            isosurface with 
        reciprocal_lattice : (3,3)
            Reciprocal lattice of the structure
        interpolation_factor : int
            number of kpoints in every direction will increase by this factor
        dxyz : (3,)
            a list of distance between each grid points in each 
            direction [dx,dy,dz]. this is used for padding 
        """





        self.kpoints = kpoints
        self.band = band
        self.spd = spd
        self.reciprocal_lattice = reciprocal_lattice
        self.fermi = fermi
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy

        self.brillouin_zone = self._get_brilloin_zone()
        

        
        Isosurface.__init__(self,
                            XYZ=self.kpoints,
                            V=self.band,
                            isovalue=self.fermi,
                            algorithm='lewiner',
                            interpolation_factor=interpolation_factor,
                            padding=None,
                            transform_matrix=self.reciprocal_lattice,
                            boundaries=self.brillouin_zone)
        if self.spd is not None and self.verts is not None:
            self.project_color(cmap, vmin, vmax)


    def project_color(self,cmap,vmin,vmax):
        if self.spd is not None:
            XYZ_extended = self.XYZ.copy()
            scalars_extended = self.spd.copy()
            for ix in range(3):

                temp = self.XYZ.copy()
                temp[:,ix]+=1
                XYZ_extended=np.append(XYZ_extended,temp,axis=0)
                scalars_extended = np.append(scalars_extended,self.spd,axis=0)
                temp = self.XYZ.copy()
                temp[:,ix]-=1
                XYZ_extended= np.append(XYZ_extended,temp,axis=0)
                scalars_extended = np.append(scalars_extended,self.spd,axis=0)

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()
            
            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
            # XYZ_transformed = XYZ_extended
            
            if self.projection_accuracy.lower()[0] == 'n': #normal
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, self.centers, method="nearest"
                    )
            elif self.projection_accuracy.lower()[0] =='h': #high
                colors = interpolate.griddata(
                        XYZ_transformed, scalars_extended, self.centers, method="linear"
                        )
            self.set_scalars(colors)
            self.set_color_with_cmap(cmap,vmin,vmax)



    def _get_brilloin_zone(self):
        return BrillouinZone(self.reciprocal_lattice)
