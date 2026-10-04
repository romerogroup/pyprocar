import itertools
import logging

import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)

_FACE_CANDIDATES = np.array([s for s in itertools.product(range(-2, 3), repeat=3) if any(s)])


def zone_face_steps(reciprocal_lattice: np.ndarray) -> np.ndarray:
    """Integer coefficients, in the rows of ``reciprocal_lattice``, of the lattice vectors
    whose bisector planes bound the first Brillouin zone.

    A candidate is dropped when its midpoint lies on or beyond the bisector plane of another.
    """
    zone = _FACE_CANDIDATES @ np.asarray(reciprocal_lattice, dtype=np.float64)
    beyond = zone @ zone.T >= (zone * zone).sum(axis=1)[:, None] * (1 - 1e-9)
    np.fill_diagonal(beyond, False)
    return _FACE_CANDIDATES[~beyond.any(axis=0)]


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

        kpoints = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec = i * self.reciprocal[0] + j * self.reciprocal[1] + k * self.reciprocal[2]
                    kpoints.append(vec)
        # print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts = brill.vertices

        return np.array(verts, dtype=float), faces

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

        kpoints = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec = i * self.reciprocal[0] + j * self.reciprocal[1] + k * self.reciprocal[2]
                    kpoints.append(vec)
        # print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts = brill.vertices

        return np.array(verts, dtype=float), faces

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
