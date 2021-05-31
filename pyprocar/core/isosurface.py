from .surface import Surface
import numpy as np
from scipy import interpolate
from skimage import measure

__author__ = "Pedram Tavadze"
__maintainer__ = "Pedram Tavadze"
__email__ = "petavazohi@mail.wvu.edu"
__date__ = "March 31, 2020"


class Isosurface(Surface):
    def __init__(
            self,
            XYZ=None,
            V=None,
            isovalue=None,
            V_matrix=None,
            algorithm='lewiner',
            interpolation_factor=1,
            padding=None,
            transform_matrix=None,
            boundaries=None,
    ):
        """
        This class contains a surface that finds all the poins correcponding 
        to the following equation
        V(X,Y,Z) = f

        Parameters
        ----------
        XYZ : TYPE, list of lists of floats, (n,3)
            DESCRIPTION. a list of coordinates [[x1,y1,z1],[x2,y2,z2],...] 
            corresponding V
        V : TYPE, list of floats, (n,)
            DESCRIPTION. a list of values [V1,V2,...] corresponding to XYZ
            XYZ[0] >>> V[0]
            XYZ[1] >>> V[1]
        isovalue : TYPE, float
            DESCRIPTION. The constant value of the surface (f)
        V_matrix : TYPE, float (nx,ny,nz) 
            DESCRIPTION. one can present V_matrix instead of XYZ and V. 
            V_matrix is a matrix representation of XYZ and V together. This 
            matrix is generated if XYZ and V are provided. 
        algorithm : TYPE, string
            DESCRIPTION. The default is 'lewiner'. The algorithm used to find the isosurface, This 
            function used scikit-image to find the isosurface. possibilities 
            ['classic','lewiner']
        interpolation_factor : TYPE, int
            DESCRIPTION. The default is 1. This module uses Fourier Transform 
            interpolation. interpolation factor will increase the grid points
            in each direction by a this factor, the dafault is set to 1
        padding : TYPE, list of float (3,)
            DESCRIPTION. padding is used for periodic datasets such as bands in
            a solid state calculation. e.g. The 1st BZ is not covered fully so
            one might want to pad the matrix with wrap(look at padding in 
            numpy for wrap), afterwards one has to clip the surface to the 
            first BZ. easily doable using pyvista of trimesh
            padding goes as follows np.pad(self.eigen_matrix,
                              ((padding[0]/2, padding[0]/2),
                              (padding[1]/2, padding[1]/2)
                              (padding[2]/2, padding[2])),
                              "wrap")
            In other words it creates a super cell withpadding
        transform_matrix : TYPE, (3,3) float
            DESCRIPTION. applies an transformation to the vertices VERTS_prime=T*VERTS
        boundaries : TYPE, pyprocar surface
            DESCRIPTION. The default is None. The boundaries in which the isosurface will be clipped with
            for example the first brillouin zone 

        """

        self.XYZ = np.array(XYZ)
        self.V = V
        self.isovalue = isovalue
        self.V_matrix = V_matrix
        self.algorithm = algorithm
        self.padding = padding
        self.interpolation_factor = interpolation_factor
        self.transform_matrix = transform_matrix
        self.boundaries = boundaries
        
        if self.algorithm not in ['classic', 'lewiner']:
            print(
                "The algorithm chose has to be from ['classic','lewiner'], automtically choosing 'lewiner'"
            )
            self.algorithm = 'lewiner'

        if self.V_matrix is None:
            self.V_matrix = map2matrix(self.XYZ, self.V)

        if self.padding is None:
            
            self.padding = [self.nX*2 // 2, self.nY*2 // 2, self.nZ*2 // 2]
        else:
            
            self.padding = [
                self.nX // 2 * padding[0], self.nY // 2 * padding[1],
                self.nZ // 2 * padding[2]
            ]
       
       

        verts, faces, normals, values = self._get_isosurface(
            interpolation_factor)

        if verts is not None and faces is not None:
            if transform_matrix is not None:
                verts = np.dot(verts,  transform_matrix)
            """
            Python, unlike statically typed languages such as Java, allows complete
            freedom when calling methods during object initialization. However, 
            standard object-oriented principles apply to Python classes using deep 
            inheritance hierarchies. Therefore the developer has responsibility for 
            ensuring that objects are properly initialized when there are multiple 
            __init__ methods that need to be called.
            For this reason I will make one temporary surface and from there I will
            using the other surface provided.
            """

            if boundaries is not None:
                suprecell_surface = Surface(verts=verts,
                                            faces=faces,
                                            face_normals=normals)
                if not np.isnan(suprecell_surface.pyvista_obj.points[0,0]):
                    verts, faces = self.clip(suprecell_surface, boundaries)
                #verts, faces = self.clip(suprecell_surface, boundaries)

        Surface.__init__(self, verts=verts, faces=faces, face_normals=normals)

    def clip(self, S1, S2):
        """
        This function clips S1 using the boundaries of S2 and returns 

        Parameters
        ----------
        S1 : TYPE pyprocar surface
            DESCRIPTION.
        S2 : TYPE pyprocar surface
            DESCRIPTION.

        Returns
        -------
        verts,faces

        """


        for iface in range(len(S2.faces)):
            normal = S2.face_normals[iface]
            
            center = np.average(S2.verts[S2.faces[iface]], axis=0)

            S1.pyvista_obj.clip(origin=center, normal=normal, inplace=True)
        

        faces = []
        courser = 0
        for i in range(S1.pyvista_obj.n_faces):
            npoints = S1.pyvista_obj.faces[courser]
            face = []
            courser += 1
            for ipoint in range(npoints):
                face.append(S1.pyvista_obj.faces[courser])
                courser += 1
            faces.append(face)
            

        return S1.pyvista_obj.points, faces

    @property
    def X(self):
        """
        

        Returns
        -------
        TYPE numpy array 
            DESCRIPTION. list of grids in X direction

        """
        return np.unique(self.XYZ[:, 0])

    @property
    def Y(self):
        """
        

        Returns
        -------
        TYPE numpy array 
            DESCRIPTION. list of grids in Y direction
        """
        return np.unique(self.XYZ[:, 1])

    @property
    def Z(self):
        """
        

        Returns
        -------
        TYPE numpy array 
            DESCRIPTION. list of grids in Z direction

        """
        return np.unique(self.XYZ[:, 2])

    @property
    def dxyz(self):
        """
        

        Returns
        -------
        list
            DESCRIPTION. length between points in each direction

        """
        dx = np.abs(self.X[-1] - self.X[-2])
        dy = np.abs(self.Y[-1] - self.Y[-2])
        dz = np.abs(self.Z[-1] - self.Z[-2])
        return [dx, dy, dz]

    @property
    def nX(self):
        """
        

        Returns
        -------
        TYPE int
            DESCRIPTION. number of points in the grid in X direction

        """
        return len(self.X)

    @property
    def nY(self):
        """
        

        Returns
        -------
        TYPE int
            DESCRIPTION. number of points in the grid in Y direction

        """
        return len(self.Y)

    @property
    def nZ(self):
        """
        

        Returns
        -------
        TYPE int
            DESCRIPTION. number of points in the grid in Z direction

        """
        return len(self.Z)

    @property
    def surface_boundaries(self):
        """
        This function tries to find the isosurface using no interpolation to find the 
        correct positions of the surface to be able to shift to the interpolated one
        to the correct position

        Returns
        -------
        list of tuples 
            DESCRIPTION. [(mins[0],maxs[0]),(mins[1],maxs[1]),(mins[2],maxs[2])]

        """

        padding_x = self.padding[0]
        padding_y = self.padding[1]
        padding_z = self.padding[2]
        
        eigen_matrix = np.pad(self.V_matrix,
                              ((padding_x, padding_x), (padding_y, padding_y),
                               (padding_z, padding_z)), "wrap")
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                eigen_matrix, self.isovalue)
            for ix in range(3):
                verts[:, ix] -= verts[:, ix].min()
                verts[:, ix] -= (verts[:, ix].max() -
                                 verts[:, ix].min()) / 2  #+self.origin[ix]
                verts[:, ix] *= self.dxyz[ix]
            mins = verts.min(axis=0)
            maxs = verts.max(axis=0)

            return [(mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])]
        except:
            return None

    def _get_isosurface(self, interp_factor=1):
        """
        

        Parameters
        ----------
        interp_factor : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION. verts
        TYPE
            DESCRIPTION. faces
        TYPE
            DESCRIPTION. normals
        TYPE
            DESCRIPTION. vlues

        """

        # Amount of kpoints needed to add on to fully sample 1st BZ

        padding_x = self.padding[0]
        padding_y = self.padding[1]
        padding_z = self.padding[2]
        
        eigen_matrix = np.pad(self.V_matrix,
                              ((padding_x, padding_x), (padding_y, padding_y),
                               (padding_z, padding_z)), "wrap")
   
        bnd = self.surface_boundaries
        
        if interp_factor != 1:
            # Fourier interpolate the mapped function E(x,y,z)

            eigen_matrix = fft_interpolate(eigen_matrix, interp_factor)

            # after the FFT we loose the center of the BZ, using numpy roll we
            # bring back the center of the BZ
            # eigen_matrix = np.roll(eigen_matrix, 4  ,
            #     axis=[0, 1, 2])

        try:
            
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                eigen_matrix, self.isovalue)
            
        except BaseException:
            print("No isosurface for this band")
            return None, None, None, None
        # recenter

        for ix in range(3):
            
            if np.any(self.XYZ >= 0.5):
                verts[:, ix] *= self.dxyz[ix] / interp_factor
                verts[:, ix] -= 1*self.supercell[ix]
                # verts[:, ix] -= 1*self.padding[ix]
 
                   
            else:
                verts[:, ix] -= verts[:, ix].min()
                verts[:, ix] -= (verts[:, ix].max() -
                                  verts[:, ix].min()) / 2
                
                verts[:, ix] *= self.dxyz[ix] / interp_factor

                if bnd is not None and interp_factor != 1:
                    verts[:, ix] -= (verts[:, ix].min() - bnd[ix][0])
                    
                    
                    
                    
                    
            #+self.origin[ix]
            # verts[:, ix] *= self.dxyz[ix] / interp_factor
            
            # print(self.dxyz)

            # if self.file == "bxsf":
            #     verts[:, ix] -= 0.5
            # if bnd is not None and interp_factor != 1:
            #     print((verts[:, ix].min() - bnd[ix][0]))
            #     verts[:, ix] -= (verts[:, ix].min() - bnd[ix][0])
            #     if self.file == "bxsf":
            #         verts[:, ix] -= 0.50
                    
                    
            #     x_shift = verts[:,0].min() - bnd[0]
            # y_shift = verts[:,1].min() - bnd[1]
            # z_shift = verts[:,2].min() - bnd[2]

        # transfare from fraction to cartesian
        # verts = np.dot(verts, self.reciprocal_)
        # new_faces = np.zeros(shape=(len(faces), 4))
        # new_faces[:, 0] = 3
        # new_faces[:, 1:] = faces
        # faces = new_faces
        return verts, faces, normals, values


def map2matrix(XYZ, V):
    """
    mapps an Irregular grid to a regular grid

    Parameters
    ----------
    XYZ : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.

    Returns
    -------
    mapped_func : TYPE
        DESCRIPTION.

    """
    XYZ = XYZ
    V = V
    
    X = np.unique(XYZ[:, 0])
    Y = np.unique(XYZ[:, 1])
    Z = np.unique(XYZ[:, 2])
    
    mapped_func = np.zeros(shape=(len(X), len(Y), len(Z)))
    #kpoint_matrix = np.zeros(shape=(len(kx), len(ky), len(kz), 3)) This was added to check if the mesh grid is working

    count = 0
    for ix in range(len(X)):
        condition1 = XYZ[:, 0] == X[ix]
        count += 1

        for iy in range(len(Y)):
            condition2 = XYZ[:, 1] == Y[iy]
            
            #print(count)
            for iz in range(len(Z)):
                
                condition3 = XYZ[:, 2] == Z[iz]
                tot_cond = np.all([condition1, condition2, condition3], axis=0)
                if len(V[tot_cond]) != 0:

                    mapped_func[ix, iy, iz] = V[tot_cond][0]
                    # kpoint_matrix[ikx, iky, ikz] = [
                    #     kx[ikx], ky[iky], kz[ikz]]
                else:
                    mapped_func[ix, iy, iz] = np.nan
                    # kpoint_matrix[ikx, iky, ikz] = [np.nan, np.nan, np.nan]
    return mapped_func


def fft_interpolate(function, interpolation_factor=2):
    """
    if I = interpolation_factor
    This function withh recieve f(x,y,z) with dimensions of (nx,ny,nz)
    and returns f(x,y,z) with dimensions of (nx*I,ny*I,nz*I)
    """

    eigen_fft = np.fft.fftn(function)
    shifted_fft = np.fft.fftshift(eigen_fft)
    nx, ny, nz = np.array(shifted_fft.shape)
    pad_x = nx * (interpolation_factor - 1) // 2
    pad_y = ny * (interpolation_factor - 1) // 2
    pad_z = nz * (interpolation_factor - 1) // 2
    new_matrix = np.pad(
        shifted_fft,
        ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)),
        "constant",
        constant_values=0,
    )
    
    new_matrix = np.fft.ifftshift(new_matrix)
    interpolated = np.real(np.fft.ifftn(new_matrix)) * (interpolation_factor**
                                                        3)

    return interpolated
