__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List
import numpy as np
from scipy import interpolate
from skimage import measure
import pyvista as pv

from .surface import Surface

class Isosurface(Surface):
    """
    This class contains a surface that finds all the points corresponding
    to the following equation. 
    V(X,Y,Z) = f

    Parameters
    ----------
    XYZ : List of lists of floats, (n,3)
        XYZ must be between (0.5,0.5]. a list of coordinates [[x1,y1,z1],[x2,y2,z2],...]
        corresponding V
    V : TYPE, list of floats, (n,)
        DESCRIPTION. a list of values [V1,V2,...] corresponding to XYZ
        XYZ[0] >>> V[0]
        XYZ[1] >>> V[1]
    isovalue : float
        The constant value of the surface (f)
    V_matrix : float (nx,ny,nz)
        One can present V_matrix instead of XYZ and V.
        V_matrix is a matrix representation of XYZ and V together. 
        This matrix is generated if XYZ and V are provided.
    algorithm : str
        The default is 'lewiner'. The algorithm used to find the isosurface, This
        function used scikit-image to find the isosurface. possibilities
        ['classic','lewiner']
    interpolation_factor : int
        The default is 1. This module uses Fourier Transform
        interpolation. interpolation factor will increase the grid points
        in each direction by a this factor, the dafault is set to 1
    padding : list of float (3,)
        Padding is used for periodic datasets such as bands in
        a solid state calculation. e.g. The 1st BZ is not covered fully so
        one might want to pad the matrix with wrap(look at padding in
        numpy for wrap), afterwards one has to clip the surface to the
        first BZ. easily doable using pyvista of trimesh
        padding goes as follows 
        
        .. code-block::
            :linenos: 
            
            np.pad(self.eigen_matrix,
                    ((padding[0]/2, padding[0]/2),
                    (padding[1]/2, padding[1]/2)
                    (padding[2]/2, padding[2])),
                    "wrap")

        In other words it creates a super cell withpadding
    transform_matrix : np.ndarray (3,3) float
        Applies an transformation to the vertices VERTS_prime=T*VERTS
    boundaries : pyprocar.core.surface
        The default is None. The boundaries in which the isosurface will be clipped with
        for example the first brillouin zone

    """

    def __init__(
            self,
            XYZ:np.ndarray,
            V:np.ndarray,
            isovalue:float,
            V_matrix=None,
            algorithm:str='lewiner',
            interpolation_factor:int=1,
            padding:List[int]=None,
            transform_matrix:np.ndarray=None,
            boundaries=None,
        ):
        

        self.XYZ = np.array(XYZ)
        self.V = V
        self.isovalue = isovalue
        self.V_matrix = V_matrix
        self.algorithm = algorithm
        self.padding = padding
        self.supercell = padding 
        self.interpolation_factor = interpolation_factor
        self.transform_matrix = transform_matrix
        self.boundaries = boundaries
        
        if self.algorithm not in ['classic', 'lewiner']:

            print(
                "The algorithm chose has to be from ['classic','lewiner'], automtically choosing 'lewiner'"
            )
            self.algorithm = "lewiner"

        if self.V_matrix is None:
            self.V_matrix = map2matrix(self.XYZ, self.V)

        if self.padding is None:
            
            self.padding = [self.nX*2 // 2, self.nY*2 // 2, self.nZ*2 // 2]

        else:
            
            self.padding = [
                self.nX // 2 * padding[0],
                self.nY // 2 * padding[1],
                self.nZ // 2 * padding[2],
            ]
        
        verts, faces, normals, values = self._get_isosurface(interpolation_factor)
        

        
        if verts is not None and faces is not None:
            if transform_matrix is not None:
                verts = np.dot(verts,  transform_matrix)
                column_of_verts_of_triangles = [3 for _ in range(len(faces[:,0])) ]
                faces = np.insert(arr = faces,obj=0, values = column_of_verts_of_triangles, axis = 1)
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
                supercell_surface = pv.PolyData(var_inp=verts, faces=faces)
                for normal,center in zip(boundaries.face_normals, boundaries.centers):
                    supercell_surface.clip(origin=center, normal=normal, inplace=True)

                if len(supercell_surface.points) == 0:
                    raise Exception("Clippping destroyed mesh.")
    
                verts = supercell_surface.points
                faces = supercell_surface.faces

 
        super().__init__(verts=verts, faces=faces)

        return None
        
     

    @property
    def X(self):
        """
        Returns the unique x values of the grid

        Returns
        -------
        np.ndarray
            list of grids in X direction

        """
        return np.unique(self.XYZ[:, 0])

    @property
    def Y(self):
        """
        Returns the unique y values of the grid

        Returns
        -------
        np.ndarray
            List of grids in Y direction
        """
        return np.unique(self.XYZ[:, 1])

    @property
    def Z(self):
        """
        Returns the unique z values of the grid

        Returns
        -------
        np.ndarray
            List of grids in Z direction

        """
        return np.unique(self.XYZ[:, 2])

    @property
    def dxyz(self):
        """
        Returns the spacings of the grid in the x,y,z directions.

        Returns
        -------
        List[float]
            Length between points in each direction

        """
        dx = np.abs(self.X[-1] - self.X[-2])
        dy = np.abs(self.Y[-1] - self.Y[-2])
        dz = np.abs(self.Z[-1] - self.Z[-2])
        return [dx, dy, dz]

    @property
    def nX(self):
        """
        Returns the number of points in the grid in X direction

        Returns
        -------
        int
            The number of points in the grid in X direction

        """
        return len(self.X)

    @property
    def nY(self):
        """
        Returns the number of points in the grid in Y direction

        Returns
        -------
        int
            The number of points in the grid in Y direction

        """
        return len(self.Y)

    @property
    def nZ(self):
        """
        Returns the number of points in the grid in Z direction

        Returns
        -------
        int
            The number of points in the grid in Z direction

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

        eigen_matrix = np.pad(
            self.V_matrix,
            ((padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z)),
            "wrap",
        )
        try:
            verts, faces, normals, values = measure.marching_cubes(
                eigen_matrix, self.isovalue
            )
            for ix in range(3):
                verts[:, ix] -= verts[:, ix].min()
                verts[:, ix] -= (
                    verts[:, ix].max() - verts[:, ix].min()
                ) / 2  # +self.origin[ix]
                verts[:, ix] *= self.dxyz[ix]
            mins = verts.min(axis=0)
            maxs = verts.max(axis=0)

            return [(mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])]
        except Exception as e:
            # print(e)
            # print("No isosurface for this band")
            return None

    def _get_isosurface(self, interp_factor:float=1):
        """
        The helper method will try to find the iso surface by using the marching cubes algorithm

        Parameters
        ----------
        interp_factor : float, optional
            Interpolation factor. The default is 1.

        Returns
        -------
        np.ndarray
            The vertices of the isosurface. verts
        np.ndarray
            The faces of the isosurface. faces
        np.ndarray
            The normals to the faces of the isosurface. normals
        np.ndarray
            The values of the isosurface. values

        """

        # Amount of kpoints needed to add on to fully sample 1st BZ

        padding_x = self.padding[0]
        padding_y = self.padding[1]
        padding_z = self.padding[2]

        eigen_matrix = np.pad(
            self.V_matrix,
            ((padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z)),
            "wrap",
        )

        bnd = self.surface_boundaries

        if interp_factor != 1:
            # Fourier interpolate the mapped function E(x,y,z)

            eigen_matrix = fft_interpolate(eigen_matrix, interp_factor)

            # after the FFT we loose the center of the BZ, using numpy roll we
            # bring back the center of the BZ
            # eigen_matrix = np.roll(eigen_matrix, 4  ,
            #     axis=[0, 1, 2])

        try:
            verts, faces, normals, values = measure.marching_cubes(
                eigen_matrix, self.isovalue
            )

        except Exception as e:
            # print(e)
            # print("No isosurface for this band")
            return None, None, None, None
            
        # recenter
        for ix in range(3):
            verts[:, ix] -= verts[:, ix].min()
            verts[:, ix] -= (verts[:, ix].max() -
                                verts[:, ix].min()) / 2
            
            verts[:, ix] *= self.dxyz[ix] / interp_factor

            if bnd is not None and interp_factor != 1:
                verts[:, ix] -= (verts[:, ix].min() - bnd[ix][0])
                    
        return verts, faces, normals, values


    
def map2matrix(XYZ, V):
    """
    Maps an Irregular grid to a regular grid

    Parameters
    ----------
    XYZ : np.ndarray
        The points of the irregular grid.
    V : np.ndarray
        The values of the irregular grid.

    Returns
    -------
    mapped_func : np.ndarray
        The points of the regular grid.

    """
    XYZ = XYZ
    V = V

    X = np.unique(XYZ[:, 0])
    Y = np.unique(XYZ[:, 1])
    Z = np.unique(XYZ[:, 2])

    mapped_func = np.zeros(shape=(len(X), len(Y), len(Z)))
    # kpoint_matrix = np.zeros(shape=(len(kx), len(ky), len(kz), 3)) This was added to check if the mesh grid is working

    count = 0
    for ix in range(len(X)):
        condition1 = XYZ[:, 0] == X[ix]
        count += 1

        for iy in range(len(Y)):
            condition2 = XYZ[:, 1] == Y[iy]

            # print(count)
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
    This method will interpolate using a Fast-Fourier Transform
    
    if I = interpolation_factor
    This function withh recieve f(x,y,z) with dimensions of (nx,ny,nz)
    and returns f(x,y,z) with dimensions of (nx*I,ny*I,nz*I)

    Parameters
    ----------
    function : np.ndarray
        The values array to do the interpolation on.
    interpolation_factor : int, optional
        Interpolation Factor, by default 2

    Returns
    -------
    np.ndarray
        The interpolated points
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
    interpolated = np.real(np.fft.ifftn(new_matrix)) * (interpolation_factor ** 3)

    return interpolated
