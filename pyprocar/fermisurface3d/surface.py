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



class Surface:
    def __init__(self,
                 verts=None,
                 faces=None,
                 face_normals=None,
                 vert_normals=None,
                 face_colors=None,
                 vert_colors=None,
                 face_vectors=None,
                 scalars=None):
        """
        Surface is a class that holds information about a surface
        To create a surface the minimum requirements are verts and faces

        Parameters
        ----------
        verts : TYPE, optional
            DESCRIPTION. The default is None. The list of verticies that create the surface.
        faces : TYPE, optional
            DESCRIPTION. The default is None. The list of connectivity between verts that create the 
            surface.
        face_normals : TYPE, optional. The list of normal vectors to each face.
            DESCRIPTION. The default is None.
        vert_normals : TYPE, optional
            DESCRIPTION. The default is None. The list of normal vectors to each vertex.
        face_colors : TYPE, optional
            DESCRIPTION. The default is None. The list of colors of each face.
        vert_colors : TYPE, optional
            DESCRIPTION. The default is None. The list of colors of each vertex.
        face_vectors : TYPE, optional
            DESCRIPTION. The default is None. The list of vectors one wants to attach to the 
            surface(glyphs) Only useful in pyvista objects

        :param nfaces: Total number of faces
        :param nverts: Total number of verts
        :param pyvista_obj: a pyvista PolyData grid class method 
        https://docs.pyvista.org/core/points.html#pyvista-polydata-grid-class-methods
        :param trimesh_obj: a trimesh Trimesh class 
        https://trimsh.org/trimesh.base.html?highlight=alpha


        Returns
        -------
        None.
        """
     
        self.verts = verts
        self.faces = faces
        self.face_normals = face_normals
        self.vert_normals = vert_normals
        self.face_colors = face_colors
        self.vert_colors = vert_colors
        self.face_vectors = face_vectors
        self.scalars = scalars
        self.pyvista_obj = None
        self.trimesh_obj = None
        self.centers = None
        if self.verts is not None and self.faces is not None:
            self._create_pyvista()
            self._create_trimesh()
            if self.face_normals is None:
                self.face_normals=self.pyvista_obj.face_normals
            if self.vert_normals is None:
                self.vert_normals=self.pyvista_obj.point_normals
            self.centers = np.zeros(shape=(len(self.faces), 3))
            for iface in range(self.nfaces):
                self.centers[iface, 0:3] = np.average(verts[faces[iface]], axis=0)
            # self.centers = self.pyvista_obj.cell_centers().points
        
    @property
    def nfaces(self):
        return len(self.faces)
    
    
    def _create_pyvista(self):
        """
        creates pyvista object this object has to have a certain order for faces
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
        if self.face_vectors is not None:
            self.pyvista_obj['vectors'] = self.face_vectors
            self.pyvista_obj.set_active_vetors('vectors')

    def _create_trimesh(self):
        """
        creates a trimesh object
        """
        
        if np.any(np.array([len(x) for x in self.faces])>3):
            faces = []
            for i in range(0,len(self.pyvista_obj.triangulate().faces),4):
                point_1 = self.pyvista_obj.triangulate().faces[i+1]
                point_2 = self.pyvista_obj.triangulate().faces[i+2]
                point_3 = self.pyvista_obj.triangulate().faces[i+3]
                faces.append([point_1, point_2, point_3])
            self.trimesh_obj = trimesh.Trimesh(vertices=self.verts,faces=faces)

        else:

            self.trimesh_obj = trimesh.Trimesh(vertices=self.verts,faces=self.faces)
        

    def set_scalars(self,
                    scalars,
                    ):
        self.scalars=scalars
        self.pyvista_obj['scalars'] = self.scalars
        self.pyvista_obj.set_active_scalars('scalars')
        


    def set_color_with_cmap(self,
                            cmap='viridis',
                            vmin=None,
                            vmax=None):
        if vmin is None:
            vmin = min(self.scalars)
        if vmax is None:
            vmax = max(self.scalars)                
        norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap)
        colors = np.array([cmap(norm(x)) for x in self.scalars]).reshape(-1,4)
        self.face_colors = colors

        # This next line will make all the surfaces double sided if you want 
        # only show one side comment the next line
        if len(self.trimesh_obj.faces) == self.nfaces:
            self.trimesh_obj.faces = np.vstack(
                (self.trimesh_obj.faces, np.fliplr(self.trimesh_obj.faces)
                 )
                )
        
        if len(self.trimesh_obj.faces) == self.nfaces:
            self.trimesh_obj.visual.face_colors = colors
        else:
            self.trimesh_obj.visual.face_colors = np.append(colors,colors,axis=0)
        


            # norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
            # cmap = cm.get_cmap(self.cmap)
            # colors = [cmap(norm(x)) for x in self.scalars]
            # self.face_colors=colors
            # if len(self.trimesh_obj.faces) == self.nfaces:
            #     self.trimesh_obj.visual.face_colors = colors
            # else:
            #     self.trimesh_obj.visual.face_colors = colors*2
            # self.pyvista_obj['scalars'] = self.scalars
            # self.pyvista_obj.set_active_scalars('scalars')
        
    def export(self,file_obj='output.glb',file_type='glb'):
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
        self.trimesh_obj.export(file_obj,file_type)

def convert_from_pyvista_faces(pyvista_obj):
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
    This functtion depends on union function from trimesh
    :param S2: The surface you want to union, it has to be an object of Surface
    :param enegine: backend engine used to calculate the union 'blender','scad','pyvista', This is checked if the engine is installed of not
    :param inplace: If inplace=True it will modify the original surcafe
    else it will return a new surface.
    """
    ret = surfaces[0].pyvista_obj.copy()
    for isurface in range(1,len(surfaces)):
        ret = ret.boolean_add(surfaces[isurface].pyvista_obj,inplace=False)
    surf = Surface(verts=ret.points,
               faces=convert_from_pyvista_faces(ret),
               face_normals=ret.face_normals,
               vert_normals=ret.point_normals,
               scalars=ret.active_scalars)
    return surf
    
    # verts = np.concatenate([x.verts for x in surfaces],axis=0)
    # faces = np.concatenate([x.faces for x in surfaces],axis=0)

    # if sum([x.scalars is not None for x in surfaces]) > len(surfaces):
    #     scalars = np.concatenate([x.scalars for x in surfaces],axis=0)
    #     vmin = min([x.vmin for x in surfaces])
    #     vmax = min([x.vmax for x in surfaces])
    # else:
    #     scalars = None
    #     vmin = None
    #     vmax = None
    
    # if sum([x.face_colors is not None for x in surfaces]) > len(surfaces):
    #     face_colors = np.concatenate([x.face_colors for x in surfaces],axis=0)
    # else:
    #     face_colors = None
    
    # return Surface(verts=verts,
    #                faces=faces,
    #                face_colors=face_colors,
    #                scalars=scalars,
    #                vmin=vmin,
    #                vmax=vmax)
    # normals = np.concatenate([x.scalars for x in surfaces],axis=0)

