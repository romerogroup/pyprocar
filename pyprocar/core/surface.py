import pyvista
import trimesh
import numpy as np
from shutil import which
from matplotlib import cm
from matplotlib import colors as mpcolors

__author__ = "Pedram Tavadze"
__maintainer__ = "Pedram Tavadze"
__email__ = "petavazohi@mail.wvu.edu"
__date__ = "March 31, 2020"


class Surface(object):
    """
        Surface is a class that holds information about a surface
        To create a surface the minimum requirements are verts and faces

        Parameters
        ----------
        verts : list of float (nverts,3)
            The list of verticies that create the surface.
        faces : list of integers (nfaces,3)
            The default is None. The list of connectivity between
            verts that create the surface.
        face_normals : list of float (nfaces,3)
            The list of normal vectors to each face.
        vert_normals : list of float (nverts,3)
            The list of normal vectors to each vertex.
        face_colors : list of tuples floats (nfaces,3)
            The list of colors of each face.
            **example**:``face_colors=[(1,0,0),(1,0.5,0),...,(1,0,0)]``
        vert_colors : list of tuples floats (nfaces,3)
            The list of colors of each vertex.
        vectors : list of floats (nfaces,3)
            The list of vectors one wants to attach to the
            surface(glyphs) Only useful in pyvista objects
        scalars : list of floats (nfaces,)
            The list of scalars for each face. This can represent
            the color using a color map
                


    """
    def __init__(self,
                 verts=None,
                 faces=None,
                 face_normals=None,
                 vert_normals=None,
                 face_colors=None,
                 vert_colors=None,
                 vectors=None,
                 scalars=None):

        self.verts = verts
        self.faces = faces
        self.face_normals = face_normals
        self.vert_normals = vert_normals
        self.face_colors = face_colors
        self.vert_colors = vert_colors
        self.vectors = vectors
        self.scalars = scalars
        self.test = None
        self.pyvista_obj = None
        self.trimesh_obj = None

        if self.verts is not None and self.faces is not None:
            self._create_pyvista()
            self._create_trimesh()
            if self.face_normals is None:
                self.face_normals = self.pyvista_obj.face_normals
            # if self.vert_normals is None:
            #     self.vert_normals=self.pyvista_obj.point_normals

    # @property
    # def mesh(self):
    #     return self.trimesh_obj

    # @property
    # def polydata(self):
    #     return self.polydata

    @property
    def centers(self):
        """
        Centers of faces
        Returns
        -------
        centers : list of floats (n,3)
            A list of centers of faces.

        """

        if self.verts is not None:
            centers = np.zeros(shape=(len(self.faces), 3))
            for iface in range(self.nfaces):
                centers[iface, 0:3] = np.average(self.verts[self.faces[iface]],
                                                 axis=0)
        else:
            centers = None
        return centers

    @property
    def nfaces(self):
        """
        Number of faces
        Returns
        -------
        int
           Number of faces in the surface.

        """
        return len(self.faces)

    @property
    def nverts(self):
        """
        Number or vertices.
        Returns
        -------
        int
            Number of verticies in in the surface.

        """
        return len(self.verts)

    @property
    def center_of_mass(self):
        """
        Center of mass of the vertices.
        Returns
        -------
        list float
            Center of mass of vertices.
        """
        return np.average(self.verts, axis=1)

    def _create_pyvista(self):
        """
        creates pyvista object this object has to have a certain order
        for faces
        example :
        [n_verts_1st_face,1st_vert,2nd_vert,...,nverts_2nd_face,1st_vert,2nd_vert,...]
        """
        verts = np.array(self.verts)
        faces = np.array(self.faces)
        new_faces = []

        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)
                
        self.pyvista_obj = pyvista.PolyData(verts, np.array(new_faces))
        if self.scalars is not None:
            self.pyvista_obj['scalars'] = self.scalars
            self.pyvista_obj.set_active_scalars('scalars')
        if self.vectors is not None:
            self.pyvista_obj['vectors'] = self.vectors
            self.pyvista_obj.set_active_vetors('vectors')

    def _create_trimesh(self):
        """
        creates a trimesh object
        """
        
        if np.any(np.array([len(x) for x in self.faces]) > 3):
            faces = []
            for i in range(0, len(self.pyvista_obj.triangulate().faces), 4):
                point_1 = self.pyvista_obj.triangulate().faces[i + 1]
                point_2 = self.pyvista_obj.triangulate().faces[i + 2]
                point_3 = self.pyvista_obj.triangulate().faces[i + 3]
                faces.append([point_1, point_2, point_3])
            self.trimesh_obj = trimesh.Trimesh(vertices=self.verts,
                                                faces=faces)

        else:

            self.trimesh_obj = trimesh.Trimesh(vertices=self.verts,
                                                faces=self.faces)

    def set_scalars(
            self,
            scalars,
    ):
        """
        Sets/Updates the scalars of the surface. Scalars represent a
        color using a color map.

        Parameters
        ----------
        scalars : list
            Scalars should be the same size as the number of faces.


        """
        self.scalars = scalars
        self.pyvista_obj['scalars'] = self.scalars
        self.pyvista_obj.set_active_scalars('scalars')

    def set_vectors(self, vectors_X, vectors_Y, vectors_Z):

        self.vectors = np.vstack([vectors_X, vectors_Y, vectors_Z]).T
        self.pyvista_obj['vectors'] = self.vectors
        # self.pyvista_obj.set_active_scalars('vectors')

        # self.pyvista_obj.vectors = self.vectors

    def set_color_with_cmap(self, cmap='viridis', vmin=None, vmax=None):
        """
        Sets colors for the trimesh object using the color map provided

        Parameters
        ----------
        cmap : TYPE, string
            DESCRIPTION. The default is 'viridis'.
        vmin : TYPE, float
            DESCRIPTION. The default is None.
        vmax : TYPE, optional
            DESCRIPTION. The default is None.


        """
        if vmin is None:
            vmin = min(self.scalars)
        if vmax is None:
            vmax = max(self.scalars)
        norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap)

        colors = np.array([cmap(norm(x)) for x in self.scalars]).reshape(-1, 4)
        self.face_colors = colors

        # This next line will make all the surfaces double sided if you want
        # only show one side comment the next line
        if len(self.trimesh_obj.faces) == self.nfaces:
            self.trimesh_obj.faces = np.vstack(
                (self.trimesh_obj.faces, np.fliplr(self.trimesh_obj.faces)))

        if len(self.trimesh_obj.faces) == self.nfaces:
            self.trimesh_obj.visual.face_colors = colors
        else:
            self.trimesh_obj.visual.face_colors = np.append(colors,
                                                            colors,
                                                            axis=0)

    def export(self, file_obj='output.glb', file_type='glb'):
        """
        This function uses the export function from trimesh

        Parameters
        ----------
        file_obj : TYPE, optional
            DESCRIPTION. The default is 'output.glb'.
        file_type : TYPE, optional
            DESCRIPTION. The default is 'glb'.

        Returns
        -------
        None.

        """
        self.trimesh_obj.export(file_obj, file_type)


def convert_from_pyvista_faces(pyvista_obj):
    """
    pyvista mesh faces are written in a 1d array, This function returns faces in 
    a conventional way. A list of lists, where each list contains integers numbers of 
    vert conections

    Parameters
    ----------
    pyvista_obj : TYPE PyVista mesh
        DESCRIPTION.

    Returns
    -------
    new_faces : TYPE list of lists
        DESCRIPTION. A list of lists, where each list contains integers numbers of 
    vert conections

    """
    new_faces = []
    courser = 0
    for iface in range(pyvista_obj.n_faces):
        start = courser + 1
        end = start + pyvista_obj.faces[courser]
        face = pyvista_obj.faces[start:end]
        courser = end
        new_faces.append(face)
    return new_faces


def boolean_add(surfaces):
    """
    This functtion uses boolean add from PyVista 

    Parameters
    ----------
    surfaces : TYPE list of pyprocar.Surface
        DESCRIPTION. 

    Returns
    -------
    surf : TYPE  pyprocar surface
        DESCRIPTION. The unionized surface from surfaces

    """
    try:
        ret = surfaces[0].pyvista_obj.copy()
        for isurface in range(1, len(surfaces)):
            ret = ret.boolean_add(surfaces[isurface].pyvista_obj, inplace=False)
    except:
        try:
            ret = surfaces[0].copy()
            for isurface in range(1, len(surfaces)):
                ret = ret.boolean_add(surfaces[isurface], inplace=False)
        except:
            print("Not a valid surface")

    surf = Surface(verts=ret.points,
                   faces=convert_from_pyvista_faces(ret),
                   face_normals=ret.face_normals,
                   vert_normals=ret.point_normals,
                   scalars=ret.active_scalars)
    return surf
