"""Brillouin zone calculations and representations."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import trimesh
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)


class Lines:
    """Line representation for Brillouin zone edges."""

    verts: npt.NDArray[np.float64] | None
    faces: npt.NDArray[np.intp] | None
    pyvista_line: pv.PolyData
    trimesh_line: Any
    connectivity: list[list[int]]

    def __init__(
        self,
        verts: npt.NDArray[np.float64] | None = None,
        faces: npt.NDArray[np.intp] | None = None,
    ) -> None:
        self.verts = verts
        self.faces = faces

        self.pyvista_line = pv.PolyData()
        self.trimesh_line = None
        self.connectivity = []

        self._get_connectivity()

    @property
    def nface(self) -> int:
        if self.faces is None:
            return 0
        return len(self.faces)

    def _get_connectivity(self) -> None:
        if self.faces is None:
            return
        for iface in range(len(self.faces)):
            self.connectivity.append(
                [int(self.faces[iface][0]), int(self.faces[iface][-1])]
            )  # to connect the 1st and last point
            for ipoint in range(len(self.faces[iface]) - 1):
                point_1 = int(self.faces[iface][ipoint])
                point_2 = int(self.faces[iface][ipoint + 1])
                self.connectivity.append([point_1, point_2])

    # def _create_pyvista(self):
    #     cell = []
    #     for iline in self.connectivity:
    #         cell.append([2, iline[0], iline[1]])
    #     self.pyvista_line.lines = cell

    def _create_trimesh(self) -> None:
        entities: list[Any] = []
        for iline in self.connectivity:
            # trimesh is untyped, access entities module via path
            path_module: Any = trimesh.path
            line_entity: Any = path_module.entities.Line(iline)
            entities.append(line_entity)

        self.trimesh_line = trimesh.path.Path3D(entities=entities, vertices=self.verts)


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

    reciprocal: npt.NDArray[np.float64]

    def __init__(
        self,
        reciprocal_lattice: npt.NDArray[np.float64],
        transformation_matrix: list[int] | None = None,
    ) -> None:
        logger.info("___Initializing BrillouinZone object___")

        self.reciprocal = reciprocal_lattice
        verts, faces = self.wigner_seitz()

        # Format faces for pv.PolyData
        new_faces: list[int] = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        # Initialize with the properly formatted faces array
        # pyvista has incomplete type stubs, use cast to suppress warning
        cast(Any, pv.PolyData.__init__)(self, verts, new_faces)

        logger.debug(f"BrillouinZone faces: {len(faces)}")
        logger.debug(f"BrillouinZone verts: {verts.shape}")

        self._fix_normals_direction()

        return None

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], self.cell_centers().points
        )
        return result

    @property
    def faces_array(self) -> list[list[int]]:
        """
        The faces listed in a list of list which contains the faces.


        Returns
        -------
        new_faces : list
            A list of faces

        """
        new_faces: list[list[int]] = []

        face: list[int] = []
        count = 0
        num_verts = 0

        pv_faces: npt.NDArray[np.intp] = cast(npt.NDArray[np.intp], self.faces)
        for iverts_in_face, verts_in_face in enumerate(pv_faces):
            if iverts_in_face == 0:
                num_verts = int(verts_in_face)
                face = [num_verts]
            else:
                if count == num_verts:
                    count = 0
                    new_faces.append(face)
                    num_verts = int(verts_in_face)
                    face = [num_verts]
                elif iverts_in_face == len(pv_faces) - 1:
                    face.append(int(verts_in_face))
                    new_faces.append(face)
                else:
                    count += 1
                    face.append(int(verts_in_face))

        return new_faces

    def wigner_seitz(self) -> tuple[npt.NDArray[np.float64], list[list[int]]]:
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """
        logger.info("___Calculating Wigner Seitz cell___")

        kpoints: list[npt.NDArray[np.float64]] = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec: npt.NDArray[np.float64] = (
                        i * self.reciprocal[0] + j * self.reciprocal[1] + k * self.reciprocal[2]
                    )
                    kpoints.append(vec)
        # print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces: list[list[int]] = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts: npt.NDArray[np.float64] = np.array(brill.vertices, dtype=np.float64)

        return verts, faces

    def _fix_normals_direction(self) -> None:
        """
        Helper method that calculates the normals of the Wigner seits cell
        """
        logger.info("___Fixing normals direction___")
        cell_centers: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], self.cell_centers().points
        )
        if len(cell_centers) == 0:
            logger.warning("___No centers found___")
            return None

        center: npt.NDArray[np.float64] = cell_centers[0]
        n1: npt.NDArray[np.float64] = center / np.linalg.norm(center)
        face_normals: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], self.face_normals
        )
        n2: npt.NDArray[np.float64] = face_normals[0]
        correction: np.floating[Any] = np.sign(np.dot(n1, n2))
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

    reciprocal: npt.NDArray[np.float64] | None

    def __init__(
        self,
        e_min: float,
        e_max: float,
        axis: int = 2,
        reciprocal_lattice: npt.NDArray[np.float64] | None = None,
        transformation_matrix: list[int] | None = None,
    ) -> None:
        self.reciprocal = reciprocal_lattice

        verts, faces = self.wigner_seitz()

        min_val: np.floating[Any] = verts[:, axis].min()
        max_val: np.floating[Any] = verts[:, axis].max()

        for vert in verts:
            vert_z = vert[axis]
            if np.isclose(vert_z, min_val, atol=1e-2):
                vert[axis] = e_min
            if np.isclose(vert_z, max_val, atol=1e-2):
                vert[axis] = e_max

        new_faces: list[int] = []
        for iface in faces:
            new_faces.append(len(iface))
            for ivert in iface:
                new_faces.append(ivert)

        # Initialize with the properly formatted faces array
        # pyvista has incomplete type stubs, use cast to suppress warning
        cast(Any, pv.PolyData.__init__)(self, verts, new_faces)

        self._fix_normals_direction()
        return None

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], self.cell_centers().points
        )
        return result

    @property
    def faces_array(self) -> list[list[int]]:
        """
        The faces listed in a list of list which contains the faces.


        Returns
        -------
        new_faces : list
            A list of faces

        """
        new_faces: list[list[int]] = []

        face: list[int] = []
        count = 0
        num_verts = 0

        pv_faces: npt.NDArray[np.intp] = cast(npt.NDArray[np.intp], self.faces)
        for iverts_in_face, verts_in_face in enumerate(pv_faces):
            if iverts_in_face == 0:
                num_verts = int(verts_in_face)
                face = [num_verts]
            else:
                if count == num_verts:
                    count = 0
                    new_faces.append(face)
                    num_verts = int(verts_in_face)
                    face = [num_verts]
                elif iverts_in_face == len(pv_faces) - 1:
                    face.append(int(verts_in_face))
                    new_faces.append(face)
                else:
                    count += 1
                    face.append(int(verts_in_face))

        return new_faces

    def wigner_seitz(self) -> tuple[npt.NDArray[np.float64], list[list[int]]]:
        """Calculates the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell

        Returns
        -------
        Tuple(n_verts,n_faces)
            Returns the wigner Seitz cell in the form of a tuple containing the verts and faces of the cell
        """
        if self.reciprocal is None:
            raise ValueError("reciprocal_lattice must be provided")

        kpoints: list[npt.NDArray[np.float64]] = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    vec: npt.NDArray[np.float64] = (
                        i * self.reciprocal[0] + j * self.reciprocal[1] + k * self.reciprocal[2]
                    )
                    kpoints.append(vec)
        # print(kpoints, self.reciprocal)
        brill = Voronoi(np.array(kpoints))
        faces: list[list[int]] = []
        for idict in brill.ridge_dict:
            if idict[0] == 13 or idict[1] == 13:
                faces.append(brill.ridge_dict[idict])

        verts: npt.NDArray[np.float64] = np.array(brill.vertices, dtype=np.float64)

        return verts, faces

    def _fix_normals_direction(self) -> None:
        """
        Helper method that calculates the normals of the Wigner seits cell
        """
        center: npt.NDArray[np.float64] = self.centers[0]
        n1: npt.NDArray[np.float64] = center / np.linalg.norm(center)
        face_normals: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], self.face_normals
        )
        n2: npt.NDArray[np.float64] = face_normals[0]
        correction: np.floating[Any] = np.sign(np.dot(n1, n2))
        if correction == -1:
            self.compute_normals(flip_normals=True, inplace=True)
        return None
