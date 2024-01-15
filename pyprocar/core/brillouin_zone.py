__author__ = "Pedram Tavadze"
__maintainer__ = "Pedram Tavadze"
__email__ = "petavazohi@mail.wvu.edu"
__date__ = "March 31, 2020"

from typing import List

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import Voronoi

from .surface import Surface

class Lines:
    def __init__(self, 
                verts:np.ndarray=None, 
                faces:np.ndarray=None):

        self.verts = verts
        self.faces = faces

        self.pyvista_line = pv.PolyData()
        self.trimesh_line = None
        self.connectivity = []
        
        
        self._get_connectivity()
        

    @property
    def nface(self):
        return len(self.faces)

    def _get_connectivity(self):
        for iface in range(len(self.faces)):
            self.connectivity.append(
                [self.faces[iface][0], self.faces[iface][-1]]
            )  # to connect the 1st and last point
            for ipoint in range(len(self.faces[iface]) - 1):
                point_1 = self.faces[ipoint]
                point_2 = self.faces[ipoint + 1]
                self.connectivity.append([point_1, point_2])

    # def _create_pyvista(self):
    #     cell = []
    #     for iline in self.connectivity:
    #         cell.append([2, iline[0], iline[1]])
    #     self.pyvista_line.lines = cell

    def _create_trimesh(self):
        
        entries = []
        for iline in self.connectivity:
            entries.append(trimesh.path.entries.Line(iline))

            self.trimesh_line = trimesh.path.path.Path(
                entries=entries, vertices=self.verts
            )

class BrillouinZone(Surface):
    """
    A Surface object with verts, faces and line representation, representing the BrillouinZone.
    This class will calculate the BrillouinZone corresponding to a reciprocal lattice.

    Parameters
    ----------
    reciprocal_lattice : np.ndarray, 
        Reciprocal lattice used to generate Brillouin zone usgin Wigner Seitz. (3,3) float
    transformation_matrix : np.ndarray
        Any transformation to be applied to the unit cell such as rotation or supercell. (3,3) float. defaults to None
            
    """
        

    def __init__(self, 
                reciprocal_lattice:np.ndarray, 
                transformation_matrix:List[int]=None
                ):
        
        self.reciprocal = reciprocal_lattice
        # for ix in range(3):
        # self.reciprocal[:,ix]*=supercell[ix]
        verts, faces = self.wigner_seitz()

        
        new_faces = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        super().__init__(verts=verts, faces = new_faces )

        # self._fix_normals_direction(n_faces = len(faces))
        self._fix_normals_direction()

        return None

    def wigner_seitz(self):
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """

        kpoints = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec = (
                        i * self.reciprocal[0]
                        + j * self.reciprocal[1]
                        + k * self.reciprocal[2]
                    )
                    kpoints.append(vec)
        #print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts = brill.vertices

        return np.array(verts,dtype = float), faces

    def _fix_normals_direction(self):
        """
        Helper method that calculates the normals of the Wigner seits cell
        """
        center = self.centers[0]
        n1 = center / np.linalg.norm(center)
        n2 = self.face_normals[0]
        correction = np.sign(np.dot(n1, n2))
        if correction == -1:
            self.compute_normals(flip_normals=True, inplace=True)
        return None

class BrillouinZone2D(Surface):
    """
    A Surface object with verts, faces and line representation, representing the BrillouinZone.
    This class will calculate the BrillouinZone corresponding to a reciprocal lattice.

    Parameters
    ----------
    e_min : float, 
        float
    e_max : float, 
        float
    reciprocal_lattice : np.ndarray, 
        Reciprocal lattice used to generate Brillouin zone usgin Wigner Seitz. (3,3) float
    transformation_matrix : np.ndarray
        Any transformation to be applied to the unit cell such as rotation or supercell. (3,3) float. defaults to None
            
    """
        

    def __init__(self, 
                 e_min,
                 e_max,
                reciprocal_lattice:np.ndarray, 
                transformation_matrix:List[int]=None
                ):
        
        self.reciprocal = reciprocal_lattice

        verts, faces = self.wigner_seitz()

        min_val=verts[:,2].min()
        max_val=verts[:,2].max()

        for vert in verts:
            vert_z=vert[2]
            if np.isclose(vert_z,min_val,atol=1e-2):
                vert[2]=e_min
            if np.isclose(vert_z,max_val,atol=1e-2):
                vert[2]=e_max

        new_faces = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        super().__init__(verts=verts, faces = new_faces )
        self._fix_normals_direction()
        return None

    def wigner_seitz(self):
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """

        kpoints = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec = (
                        i * self.reciprocal[0]
                        + j * self.reciprocal[1]
                        + k * self.reciprocal[2]
                    )
                    kpoints.append(vec)
        #print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts = brill.vertices

        return np.array(verts,dtype = float), faces

    def _fix_normals_direction(self):
        """
        Helper method that calculates the normals of the Wigner seits cell
        """
        center = self.centers[0]
        n1 = center / np.linalg.norm(center)
        n2 = self.face_normals[0]
        correction = np.sign(np.dot(n1, n2))
        if correction == -1:
            self.compute_normals(flip_normals=True, inplace=True)
        return None

    