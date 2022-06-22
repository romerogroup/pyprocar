__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from shutil import which

import pyvista
# import trimesh
import numpy as np
from matplotlib import cm
from matplotlib import colors as mpcolors


# TODO add python typing to all of the functions
# TODO add trimesh 
# TODO what is the point of self.test?
# TODO uncomment self.polydata and self.mesh
# TODO add an __str__ method
# TODO change boolean add method to __add__ method and get rid of try and expect

class Surface(pyvista.PolyData):
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
    def __init__(
        self,
        verts:np.ndarray=None,
        faces:np.ndarray=None,
        face_normals:np.ndarray=None,
        vert_normals:np.ndarray=None,
        face_colors:np.ndarray=None,
        vert_colors:np.ndarray=None,
        vectors:np.ndarray=None,
        scalars:np.ndarray=None,
        ):
        
      
        super().__init__( var_inp = verts, faces = np.array(faces))
        
        # print(faces)
        # super().__init__( var_inp = verts, faces = np.array(faces))
        
        # self.face_normals = face_normals
        # self.vert_normals = vert_normals
        self.face_colors = face_colors
        self.vert_colors = vert_colors
        # self.vectors = vectors
        self.scalars = scalars
        

        self.trimesh_obj = None

        # if self.verts is not None and self.faces is not None:
        #     self._create_trimesh()
            # if self.face_normals is None:
            #     if self.pyvista_obj.face_normals is not None:
            #         self.face_normals = self.pyvista_obj.face_normals
            # if self.vert_normals is None:
            #     self.vert_normals=self.pyvista_obj.point_normals



    @property
    def centers(self):
        """
        Centers of faces
        Returns
        -------
        centers : list of floats (n,3)
            A list of centers of faces.

        """
        return self.cell_centers().points
    
    @property
    def faces_array(self):
        """
        The faces listed in a list of list which contains the faces.
        
        
        Returns
        -------


        """
        new_faces = [] 
        
        face = []
        count = 0
        
        for iverts_in_face,verts_in_face in enumerate(self.faces):
            if iverts_in_face == 0:
                num_verts = verts_in_face
                face = [num_verts]
            else:
        
                if count == num_verts:
                    count = 0
                    new_faces.append(face)
                    num_verts = verts_in_face
                    face = [num_verts]
                elif iverts_in_face == len(self.faces)-1:
                    face.append(verts_in_face)
                    new_faces.append(face)
                else:
                    count += 1
                    face.append(verts_in_face)

        return new_faces 
    

    # def _create_trimesh(self):
    #     """
    #     creates a trimesh object
    #     """
    #     # if np.any(np.array([len(x) for x in self.faces]) > 3):
    #     #     faces = []
    #     #     for i in range(0, len(self.pyvista_obj.triangulate().faces), 4):
    #     #         point_1 = self.pyvista_obj.triangulate().faces[i + 1]
    #     #         point_2 = self.pyvista_obj.triangulate().faces[i + 2]
    #     #         point_3 = self.pyvista_obj.triangulate().faces[i + 3]
    #     #         faces.append([point_1, point_2, point_3])
    #     #     self.trimesh_obj = trimesh.Trimesh(vertices=self.verts, faces=faces)

    #     # else:
    #     self.trimesh_obj = trimesh.Trimesh(vertices=self.points, faces=self.faces)

    def set_scalars(
        self,
        scalars: np.ndarray,
        scalar_name: str="scalars"
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
        self[scalar_name] = self.scalars
        # self.set_active_scalars(scalar_name)
        # self.set_active_scalars("scalars")

    def set_vectors(self, 
                    vectors_X:np.ndarray, 
                    vectors_Y:np.ndarray, 
                    vectors_Z:np.ndarray,
                    vectors_name: str="vectors"):
        def mag(vectors):
            return np.array([(vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5 for vector in vectors])
        vectors = np.vstack([vectors_X, vectors_Y, vectors_Z]).T
        
        
        self[vectors_name] = vectors
        
        self[vectors_name + "_magnitude"] = mag(vectors)
        # self.set_active_scalars('vectors')

    def set_color_with_cmap(self, 
                            cmap:str="viridis", 
                            vmin:float=None, 
                            vmax:float=None):
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
        if len(self.trimesh_obj.faces) == self.n_faces:
            self.trimesh_obj.faces = np.vstack(
                (self.trimesh_obj.faces, np.fliplr(self.trimesh_obj.faces))
            )

        if len(self.trimesh_obj.faces) == self.n_faces:
            self.trimesh_obj.visual.face_colors = colors
        else:
            self.trimesh_obj.visual.face_colors = np.append(colors, colors, axis=0)

    def export(self, 
                file_obj:str="output.glb", 
                file_type:str="glb"):
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

    # ret = surfaces[0].pyvista_obj.copy()
    # for isurface in range(1, len(surfaces)):
    #     ret = ret.boolean_add(surfaces[isurface].pyvista_obj, inplace=False)
    # surf = Surface(
    #     verts=ret.points,
    #     faces=convert_from_pyvista_faces(ret),
    #     face_normals=ret.face_normals,
    #     vert_normals=ret.point_normals,
    #     scalars=ret.active_scalars,
    # )

    return surf
