import itertools
import logging

import numpy as np
import pyvista as pv
import spglib
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)

_FACE_CANDIDATES = np.array([s for s in itertools.product(range(-2, 3), repeat=3) if any(s)])
_NEIGHBOURS = np.array(list(itertools.product(range(-1, 2), repeat=3)))
_ORIGIN = len(_NEIGHBOURS) // 2


def zone_face_steps(reciprocal_lattice: np.ndarray) -> np.ndarray:
    """Integer coefficients, in the rows of ``reciprocal_lattice``, of the lattice vectors
    whose bisector planes bound the first Brillouin zone.

    The candidates are the 5^3 block of the Delaunay-reduced basis, whatever basis is given.
    The face vectors are the 7 pairs of an obtuse superbase and the reduced basis is 3 of
    them, so each face vector has coefficients -2 to 2 in it. A candidate is dropped when its
    midpoint lies on or beyond the bisector plane of another.
    """
    lattice = np.asarray(reciprocal_lattice, dtype=np.float64)
    unit = lattice / abs(np.linalg.det(lattice)) ** (1 / 3)
    reduced = spglib.delaunay_reduce(unit)
    if reduced is None:
        raise ValueError(f"spglib cannot Delaunay-reduce the reciprocal lattice {lattice.tolist()}")
    candidates = _FACE_CANDIDATES @ np.rint(reduced @ np.linalg.inv(unit)).astype(int)
    zone = candidates @ lattice
    beyond = zone @ zone.T >= (zone * zone).sum(axis=1)[:, None] * (1 - 1e-9)
    np.fill_diagonal(beyond, False)
    return candidates[~beyond.any(axis=0)]


def _wigner_seitz(reciprocal_lattice: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Vertices and faces of the origin's Voronoi cell."""
    lattice = np.asarray(reciprocal_lattice, dtype=np.float64)
    face_steps = zone_face_steps(lattice)
    steps = np.vstack([_NEIGHBOURS, face_steps[np.abs(face_steps).max(axis=1) > 1]])
    brill = Voronoi(steps @ lattice)
    cell = [brill.ridge_dict[pair] for pair in brill.ridge_dict if _ORIGIN in pair]
    used = np.unique(np.concatenate(cell))
    return brill.vertices[used], [np.searchsorted(used, face).tolist() for face in cell]


class BrillouinZone(pv.PolyData):
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

    def __init__(self, reciprocal_lattice: np.ndarray, transformation_matrix: list[int] = None):
        logger.info("___Initializing BrillouinZone object___")

        self.reciprocal = reciprocal_lattice
        verts, faces = self.wigner_seitz()

        # Format faces for pv.PolyData
        new_faces = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        # Initialize with the properly formatted faces array
        super().__init__(verts, new_faces)

        logger.debug(f"BrillouinZone faces: {len(faces)}")
        logger.debug(f"BrillouinZone verts: {verts.shape}")

        self._fix_normals_direction()

        return None

    @property
    def centers(self):
        return self.cell_centers().points

    @property
    def faces_array(self):
        """
        The faces listed in a list of list which contains the faces.


        Returns
        -------
        new_faces : list
            A list of faces

        """
        new_faces = []

        face = []
        count = 0

        for iverts_in_face, verts_in_face in enumerate(self.faces):
            if iverts_in_face == 0:
                num_verts = verts_in_face
                face = [num_verts]
            else:
                if count == num_verts:
                    count = 0
                    new_faces.append(face)
                    num_verts = verts_in_face
                    face = [num_verts]
                elif iverts_in_face == len(self.faces) - 1:
                    face.append(verts_in_face)
                    new_faces.append(face)
                else:
                    count += 1
                    face.append(verts_in_face)

        return new_faces

    def wigner_seitz(self):
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """
        logger.info("___Calculating Wigner Seitz cell___")
        return _wigner_seitz(self.reciprocal)

    def _fix_normals_direction(self):
        """
        Helper method that calculates the normals of the Wigner seits cell
        """
        logger.info("___Fixing normals direction___")
        cell_centers = self.cell_centers().points
        if len(cell_centers) == 0:
            logger.warning("___No centers found___")
            return None

        center = cell_centers[0]
        n1 = center / np.linalg.norm(center)
        n2 = self.face_normals[0]
        correction = np.sign(np.dot(n1, n2))
        if correction == -1:
            self.compute_normals(flip_normals=True, inplace=True)
        return None


class BrillouinZone2D(pv.PolyData):
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

    def __init__(
        self,
        e_min,
        e_max,
        axis: int = 2,
        reciprocal_lattice: np.ndarray = None,
        transformation_matrix: list[int] = None,
    ):
        self.reciprocal = reciprocal_lattice

        verts, faces = self.wigner_seitz()

        min_val = verts[:, axis].min()
        max_val = verts[:, axis].max()

        for vert in verts:
            vert_z = vert[axis]
            if np.isclose(vert_z, min_val, atol=1e-2):
                vert[axis] = e_min
            if np.isclose(vert_z, max_val, atol=1e-2):
                vert[axis] = e_max

        new_faces = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        # Initialize with the properly formatted faces array
        super().__init__(verts, new_faces)

        self._fix_normals_direction()
        return None

    @property
    def centers(self):
        return self.cell_centers().points

    @property
    def faces_array(self):
        """
        The faces listed in a list of list which contains the faces.


        Returns
        -------
        new_faces : list
            A list of faces

        """
        new_faces = []

        face = []
        count = 0

        for iverts_in_face, verts_in_face in enumerate(self.faces):
            if iverts_in_face == 0:
                num_verts = verts_in_face
                face = [num_verts]
            else:
                if count == num_verts:
                    count = 0
                    new_faces.append(face)
                    num_verts = verts_in_face
                    face = [num_verts]
                elif iverts_in_face == len(self.faces) - 1:
                    face.append(verts_in_face)
                    new_faces.append(face)
                else:
                    count += 1
                    face.append(verts_in_face)

        return new_faces

    def wigner_seitz(self):
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """
        return _wigner_seitz(self.reciprocal)

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
