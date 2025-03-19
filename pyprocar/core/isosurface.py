__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
from typing import List

import numpy as np
import pyvista as pv
from skimage import measure

from .surface import Surface

logger = logging.getLogger(__name__)


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
        XYZ: np.ndarray,
        isovalue: float = 0.0,
        V: np.ndarray = None,
        V_matrix=None,
        algorithm: str = "lewiner",
        interpolation_factor: int = 1,
        padding: List[int] = None,
        transform_matrix: np.ndarray = None,
        boundaries=None,
    ):
        logger.info(f"____ Initializing Isosurface ____")
        logger.debug(f"XYZ shape: {np.array(XYZ).shape}")
        logger.debug(f"isovalue: {isovalue}")
        if V is not None:
            logger.debug(f"V shape: {V.shape}")
        if V_matrix is not None:
            logger.debug(f"V_matrix shape: {V_matrix.shape}")
        logger.debug(f"algorithm: {algorithm}")
        logger.debug(f"interpolation_factor: {interpolation_factor}")
        logger.debug(f"padding: {padding}")
        if transform_matrix is not None:
            logger.debug(f"transform_matrix shape: {transform_matrix.shape}")
        if boundaries is not None:
            logger.debug(f"boundaries: \n {boundaries}")
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
        self.algorithm = self._get_algorithm(self.algorithm)

        # print()

        if self.V_matrix is None:
            self.V_matrix = map2matrix(self.XYZ, self.V)

        self.padding = self._get_padding(self.nX, self.nY, self.nZ)

        verts, faces, normals, values = self._get_isosurface(interpolation_factor)
        verts, faces = self._process_isosurface(verts, faces)

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
        logger.debug(f"____ Getting surface boundaries ____")

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

    def _get_algorithm(self, algorithm):
        """
        This method will return the algorithm for the surface

        Returns
        -------
        algorithm : str
            The algorithm for the surface
        """
        if algorithm not in ["classic", "lewiner"]:
            print(
                "The algorithm chose has to be from ['classic','lewiner'], automtically choosing 'lewiner'"
            )
            algorithm = "lewiner"
        return algorithm

    def _get_padding(self, n_x, n_y, n_z):
        """
        This method will return the padding for the surface

        Parameters
        ----------
        n_x : int
            The number of points in the x direction
        n_y : int
            The number of points in the y direction
        n_z : int
            The number of points in the z direction

        Returns
        -------
        padding : List[int]
            The padding for the surface
        """
        if self.padding is None:
            padding = [n_x * 2 // 2, n_y * 2 // 2, n_z * 2 // 2]
        else:
            padding = [
                n_x // 2 * self.padding[0],
                n_y // 2 * self.padding[1],
                n_z // 2 * self.padding[2],
            ]
        return padding

    def _apply_transform_matrix(self, transform_matrix, verts, faces):
        """
        This method will apply the transform matrix to the surface


        Parameters
        ----------
        transform_matrix : np.ndarray
            The transform matrix
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface

        Returns
        -------
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface
        """
        if transform_matrix is not None:
            logger.debug(f"____ Applying transform matrix ____")
            verts = np.dot(verts, transform_matrix)
            column_of_verts_of_triangles = [3 for _ in range(len(faces[:, 0]))]
            faces = np.insert(
                arr=faces, obj=0, values=column_of_verts_of_triangles, axis=1
            )
        return verts, faces

    def _apply_boundaries(self, boundaries, verts, faces):
        """
        This method will apply the boundaries to the surface

        Parameters
        ----------
        boundaries : pyprocar.core.surface
            The boundaries of the surface
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface

        Returns
        -------
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface
        """
        if boundaries is not None:
            logger.debug(f"____ Applying boundaries ____")
            supercell_surface = pv.PolyData(var_inp=verts, faces=faces)
            for normal, center in zip(boundaries.face_normals, boundaries.centers):
                supercell_surface.clip(origin=center, normal=normal, inplace=True)

            if len(supercell_surface.points) == 0:
                raise Exception("Clippping destroyed mesh.")

            verts = supercell_surface.points
            faces = supercell_surface.faces

        return verts, faces

    def _process_isosurface(self, verts, faces):
        """
        This method will process the isosurface

        Parameters
        ----------
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface

        Returns
        -------
        verts : np.ndarray
            The vertices of the surface
        faces : np.ndarray
            The faces of the surface
        """
        logger.debug(f"____ Processing isosurface ____")
        if verts is not None and faces is not None:
            verts, faces = self._apply_transform_matrix(
                self.transform_matrix, verts, faces
            )
            verts, faces = self._apply_boundaries(self.boundaries, verts, faces)

        return verts, faces

    def _get_isosurface(self, interp_factor: float = 1):
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
            logger.debug(f"____ Interpolating isosurface ____")
            # Fourier interpolate the mapped function E(x,y,z)

            eigen_matrix = fft_interpolate(eigen_matrix, interp_factor)

            # after the FFT we loose the center of the BZ, using numpy roll we
            # bring back the center of the BZ
            # eigen_matrix = np.roll(eigen_matrix, 4, axis=[0, 1, 2])

        try:
            logger.debug(f"____ Applying marching cubes ____")
            logger.debug(f"eigen_matrix shape: {eigen_matrix.shape}")
            logger.debug(f"isovalue: {self.isovalue}")
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
            verts[:, ix] -= (verts[:, ix].max() - verts[:, ix].min()) / 2

            verts[:, ix] *= self.dxyz[ix] / interp_factor

            if bnd is not None and interp_factor != 1:
                verts[:, ix] -= verts[:, ix].min() - bnd[ix][0]

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
    logger.debug(f"____ Mapping irregular grid to matrix to regular grid ____")
    XYZ = XYZ
    V = V

    X = np.unique(XYZ[:, 0])
    Y = np.unique(XYZ[:, 1])
    Z = np.unique(XYZ[:, 2])

    logger.debug(f"X shape: {X.shape}")
    logger.debug(f"Y shape: {Y.shape}")
    logger.debug(f"Z shape: {Z.shape}")

    logger.debug(f"V shape: {V.shape}")

    mapped_func = np.zeros(shape=(len(X), len(Y), len(Z)))

    # print(np.unique(X))
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
    This function will receive f(x,y,z) with dimensions of (nx,ny,nz)
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
    # Handle NaN values if present
    has_nan = np.isnan(function).any()
    if has_nan:
        # Replace NaN with zeros for FFT
        function_copy = np.nan_to_num(function, nan=0.0)
    else:
        function_copy = function.copy()

    # Perform FFT
    eigen_fft = np.fft.fftn(function_copy)

    # Get dimensions of the input array
    nx, ny, nz = function_copy.shape

    # Create larger output array filled with zeros
    new_shape = (
        nx * interpolation_factor,
        ny * interpolation_factor,
        nz * interpolation_factor,
    )
    new_fft = np.zeros(new_shape, dtype=complex)

    # Calculate half-dimensions for proper frequency component placement
    nx_half = nx // 2
    ny_half = ny // 2
    nz_half = nz // 2

    # Copy low frequency components to the new array
    # Positive frequencies
    new_fft[:nx_half, :ny_half, :nz_half] = eigen_fft[:nx_half, :ny_half, :nz_half]
    new_fft[:nx_half, :ny_half, -nz_half:] = eigen_fft[:nx_half, :ny_half, -nz_half:]
    new_fft[:nx_half, -ny_half:, :nz_half] = eigen_fft[:nx_half, -ny_half:, :nz_half]
    new_fft[:nx_half, -ny_half:, -nz_half:] = eigen_fft[:nx_half, -ny_half:, -nz_half:]
    new_fft[-nx_half:, :ny_half, :nz_half] = eigen_fft[-nx_half:, :ny_half, :nz_half]
    new_fft[-nx_half:, :ny_half, -nz_half:] = eigen_fft[-nx_half:, :ny_half, -nz_half:]
    new_fft[-nx_half:, -ny_half:, :nz_half] = eigen_fft[-nx_half:, -ny_half:, :nz_half]
    new_fft[-nx_half:, -ny_half:, -nz_half:] = eigen_fft[
        -nx_half:, -ny_half:, -nz_half:
    ]

    # Perform inverse FFT to get the interpolated result
    interpolated = np.real(np.fft.ifftn(new_fft)) * interpolation_factor**3

    return interpolated
