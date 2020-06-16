import numpy as np
from scipy.spatial import Voronoi
from .surface import Surface
import pyvista as pv
import trimesh

__author__ = "Pedram Tavadze"
__maintainer__ = "Pedram Tavadze"
__email__ = "petavazohi@mail.wvu.edu"
__date__ = "March 31, 2020"

class Lines:
    import trimesh
    import pyvista as pv
    def __init__(self,verts=None,faces=None):
        
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
        for iface in range(self.nface):
            self.connectivity.append([
                    self.faces[iface][0],
                    self.faces[iface][-1]]) # to connect the 1st and last point
            for ipoint in range(len(self.faces[iface])-1):
                point_1 = self.faces[ipoint]
                point_2 = self.faces[ipoint+1]
                self.connectivity.append([point_1,point_2])
        
    def _create_pyvista(self):
        cell = []
        for iline in self.connectivity:
            cell.append([2,iline[0],iline[1]])
        self.pyvista_line.lines = cell
        
    def _create_trimesh(self):
        entries = []
        for iline in self.connectivity:
            entries.append(trimesh.path.entries.Line(iline))

        self.trimesh_line = trimesh.path.path.Path(entries=entries,
                                                   vertices=self.verts)

class BrillouinZone(Surface):
    """
    A Surface object with verts, faces and line representation, representing
    the BrillouinZone
    """
    def __init__(self,reciprocal_lattice):
        """
        Parameters
        ----------
        reciprocal_lattice : (3,3) float
            Reciprocal Lattice

        """

        self.reciprocal = reciprocal_lattice
        verts,faces = self.wigner_seitz()
        
        Surface.__init__(self,verts=verts,faces=faces)
        # self.pyvista_obj['scalars'] = [0]*len(faces)
        # self.pyvista_obj.set_active_scalars('scalars')
        self.lines = Lines(verts,faces)
        

    
    def wigner_seitz(self):
        """
        
        Returns
        -------
        TYPE
            Using the Wigner-Seitz Method, this function finds the 1st 
            Brillouin Zone in terms of vertices and faces 
        """
        
        kpoints = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec = i * self.reciprocal[0] + j * \
                        self.reciprocal[1] + k * self.reciprocal[2]
                    kpoints.append(vec)
        brill = Voronoi(np.array(kpoints))
        faces = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])
                
        verts = brill.vertices
        return np.array(verts), np.array(faces)


                