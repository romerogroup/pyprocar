"""Procar symmetry operations."""

from __future__ import annotations

import logging
from typing import TextIO

import numpy as np
import numpy.typing as npt


class ProcarSymmetry:
    """Class for applying symmetry operations to k-points and projected band data."""

    kpoints: npt.NDArray[np.float64]
    bands: npt.NDArray[np.float64]
    character: npt.NDArray[np.float64]
    sx: npt.NDArray[np.float64]
    sy: npt.NDArray[np.float64]
    sz: npt.NDArray[np.float64]
    log: logging.Logger
    ch: logging.StreamHandler[TextIO]

    def __init__(
        self,
        kpoints: npt.NDArray[np.float64],
        bands: npt.NDArray[np.float64],
        character: npt.NDArray[np.float64] | None = None,
        sx: npt.NDArray[np.float64] | None = None,
        sy: npt.NDArray[np.float64] | None = None,
        sz: npt.NDArray[np.float64] | None = None,
        loglevel: int = logging.WARNING,
    ) -> None:
        self.log = logging.getLogger("ProcarSymmetry")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(logging.Formatter("%(name)s::%(levelname)s: %(message)s"))
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        self.log.debug("ProcarSymmetry.__init__: ...")

        self.kpoints = kpoints
        self.bands = bands
        # optional arguments when not given will False, but they can still
        # be treated like arrays
        self.character = np.array([], dtype=np.float64)
        if character is not None:
            self.character = character
        self.sx = np.array([], dtype=np.float64)
        if sx is not None:
            self.sx = sx
        self.sy = np.array([], dtype=np.float64)
        if sy is not None:
            self.sy = sy
        self.sz = np.array([], dtype=np.float64)
        if sz is not None:
            self.sz = sz

        self.log.info("Kpoints : " + str(self.kpoints.shape))
        self.log.info("bands   : " + str(self.bands.shape))
        self.log.info("character  : " + str(self.character.shape))
        self.log.info("sx      : " + str(self.sx.shape))
        self.log.info("sy      : " + str(self.sy.shape))
        self.log.info("sz      : " + str(self.sz.shape))
        self.log.debug("ProcarSymmetry.__init__: ...Done")

        return

    def _q_mult(
        self,
        q1: npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        q2: npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        Multiplication of quaternions, it doesn't fit in any other place
        """
        w1: npt.NDArray[np.float64] = np.asarray(q1[0], dtype=np.float64)
        x1: npt.NDArray[np.float64] = np.asarray(q1[1], dtype=np.float64)
        y1: npt.NDArray[np.float64] = np.asarray(q1[2], dtype=np.float64)
        z1: npt.NDArray[np.float64] = np.asarray(q1[3], dtype=np.float64)
        w2: npt.NDArray[np.float64] = np.asarray(q2[0], dtype=np.float64)
        x2: npt.NDArray[np.float64] = np.asarray(q2[1], dtype=np.float64)
        y2: npt.NDArray[np.float64] = np.asarray(q2[2], dtype=np.float64)
        z2: npt.NDArray[np.float64] = np.asarray(q2[3], dtype=np.float64)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        result: npt.NDArray[np.float64] = np.array((w, x, y, z))
        return result

    def general_rotation(
        self,
        angle: float,
        rotAxis: list[float] | str | None = None,
        store: bool = True,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Apply a rotation defined by an angle and an axis.

        Returning value: (Kpoints, sx,sy,sz), the rotated Kpoints and spin
                         vectors (if not the case, they will be empty
                         arrays).

        Arguments
        angle: the rotation angle, must be in degrees!

        rotAxis : a fixed Axis when applying the symmetry, usually it is
        from Gamma to another point). It doesn't need to be normalized.
        The RotAxis can be:
           [x,y,z] : a cartesian vector in k-space.
           'x': [1,0,0], a rotation in the yz plane.
           'y': [0,1,0], a rotation in the zx plane.
           'z': [0,0,1], a rotation in the xy plane

        """
        rotAxis_arr: npt.NDArray[np.float64]
        if rotAxis is None:
            rotAxis_arr = np.array([0.0, 0.0, 1.0])
        elif rotAxis == "x" or rotAxis == "X":
            rotAxis_arr = np.array([1.0, 0.0, 0.0])
        elif rotAxis == "y" or rotAxis == "Y":
            rotAxis_arr = np.array([0.0, 1.0, 0.0])
        elif rotAxis == "z" or rotAxis == "Z":
            rotAxis_arr = np.array([0.0, 0.0, 1.0])
        else:
            rotAxis_arr = np.array(rotAxis, dtype=np.float64)
        self.log.debug("rotAxis : " + str(rotAxis_arr))
        rotAxis_arr = rotAxis_arr / np.linalg.norm(rotAxis_arr)
        self.log.debug("rotAxis Normalized : " + str(rotAxis_arr))
        self.log.debug("Angle : " + str(angle))
        angle_rad = angle * np.pi / 180
        # defining a quaternion for rotatoin
        angle_half = angle_rad / 2
        rotAxis_arr = np.asarray(rotAxis_arr * np.sin(angle_half), dtype=np.float64)
        qRot: npt.NDArray[np.float64] = np.array(
            (np.cos(angle_half), rotAxis_arr[0], rotAxis_arr[1], rotAxis_arr[2])
        )
        qRotI: npt.NDArray[np.float64] = np.array(
            (np.cos(angle_half), -rotAxis_arr[0], -rotAxis_arr[1], -rotAxis_arr[2])
        )
        self.log.debug("Rot. quaternion : " + str(qRot))
        self.log.debug("Rot. quaternion conjugate : " + str(qRotI))
        # converting self.kpoints into quaternions
        w: npt.NDArray[np.float64] = np.zeros((len(self.kpoints), 1))
        qvectors: npt.NDArray[np.float64] = np.column_stack((w, self.kpoints)).transpose()
        self.log.debug("Kpoints-> quaternions (transposed):\n" + str(qvectors.transpose()))
        qvectors = self._q_mult(qRot, qvectors)
        qvectors = self._q_mult(qvectors, qRotI).transpose()
        kpoints: npt.NDArray[np.float64] = qvectors[:, 1:]
        self.log.debug("Rotated kpoints :\n" + str(qvectors))

        # rotating the spin vector (if exist)
        sxShape, syShape, szShape = self.sx.shape, self.sy.shape, self.sz.shape
        self.log.debug("Spin vector Shapes : " + str((sxShape, syShape, szShape)))
        # The first entry has to be an array of 0s, w could do the work,
        # but if len(self.sx)==0 qvectors will have a non-defined length
        qvectors_spin: tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ] = (
            0 * self.sx.flatten(),
            self.sx.flatten(),
            self.sy.flatten(),
            self.sz.flatten(),
        )
        self.log.debug("Spin vector quaternions: \n" + str(qvectors_spin))
        qvectors_rotated = self._q_mult(qRot, qvectors_spin)
        qvectors_rotated = self._q_mult(qvectors_rotated, qRotI)
        self.log.debug("Spin quaternions after rotation:\n" + str(qvectors_rotated))
        qr1: npt.NDArray[np.float64] = np.asarray(qvectors_rotated[1], dtype=np.float64)
        qr2: npt.NDArray[np.float64] = np.asarray(qvectors_rotated[2], dtype=np.float64)
        qr3: npt.NDArray[np.float64] = np.asarray(qvectors_rotated[3], dtype=np.float64)
        sx: npt.NDArray[np.float64] = qr1.reshape(sxShape)
        sy: npt.NDArray[np.float64] = qr2.reshape(syShape)
        sz: npt.NDArray[np.float64] = qr3.reshape(szShape)

        if store is True:
            self.kpoints, self.sx, self.sy, self.sz = kpoints, sx, sy, sz
        self.log.debug("GeneralRotation: ...Done")
        return (kpoints, sx, sy, sz)

    def rot_symmetry_z(self, order: int) -> None:
        """Applies the given rotational crystal symmetry to the current
        system. ie: to unfold the irreductible BZ to the full BZ.

        Only rotations along z-axis are performed, you can use
        self.GeneralRotation first.

        The user is responsible of provide a useful input. The method
        doesn't check the physics.

        """
        self.log.debug("RotSymmetryZ:...")
        rotations = [self.general_rotation(360 * i / order, store=False) for i in range(order)]
        rotations_unzipped = list(zip(*rotations))
        self.log.debug("self.kpoints.shape (before concat.): " + str(self.kpoints.shape))
        self.kpoints = np.concatenate(rotations_unzipped[0], axis=0)
        self.log.debug("self.kpoints.shape (after concat.): " + str(self.kpoints.shape))
        self.sx = np.concatenate(rotations_unzipped[1], axis=0)
        self.sy = np.concatenate(rotations_unzipped[2], axis=0)
        self.sz = np.concatenate(rotations_unzipped[3], axis=0)
        # the bands and proj. character also need to be enlarged
        bandsChar: list[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = [(self.bands, self.character) for _ in range(order)]
        bandsChar_unzipped = list(zip(*bandsChar))
        self.bands = np.concatenate(bandsChar_unzipped[0], axis=0)
        self.character = np.concatenate(bandsChar_unzipped[1], axis=0)
        self.log.debug("RotSymmZ:...Done")

        return

    def mirror_x(self) -> None:
        """Applies the given rotational crystal symmetry to the current
        system. ie: to unfold the irreductible BZ to the full BZ.

        """
        self.log.debug("Mirror:...")
        newK: npt.NDArray[np.float64] = self.kpoints * np.array([1, -1, 1])
        self.kpoints = np.concatenate((self.kpoints, newK), axis=0)
        self.log.debug("self.kpoints.shape (after concat.): " + str(self.kpoints.shape))
        newSx: npt.NDArray[np.float64] = -1 * self.sx
        newSy: npt.NDArray[np.float64] = 1 * self.sy
        newSz: npt.NDArray[np.float64] = 1 * self.sz
        self.sx = np.concatenate((self.sx, newSx), axis=0)
        self.sy = np.concatenate((self.sy, newSy), axis=0)
        self.sz = np.concatenate((self.sz, newSz), axis=0)
        print("self.sx", self.sx.shape)
        print("self.sy", self.sy.shape)
        print("self.sz", self.sz.shape)
        # the bands and proj. character also need to be enlarged
        self.bands = np.concatenate((self.bands, self.bands), axis=0)
        self.character = np.concatenate((self.character, self.character), axis=0)
        print("self.character", self.character.shape)
        print("self.bands", self.bands.shape)
        self.log.debug("Mirror:...Done")

        return

    def translate(self, newOrigin: npt.NDArray[np.float64] | list[float]) -> None:
        """Centers the Kpoints at newOrigin, newOrigin is either and index (of
        some Kpoint) or the cartesian coordinates of one point in the
        reciprocal space.

        """
        self.log.debug("Translate():  ...")
        newOrigin_arr: npt.NDArray[np.float64]
        if len(newOrigin) == 1:
            newOrigin_idx = int(newOrigin[0])
            newOrigin_arr = np.asarray(self.kpoints[newOrigin_idx], dtype=np.float64)
        else:
            # Make sure newOrigin is a numpy array
            newOrigin_arr = np.array(newOrigin, dtype=np.float64)
        self.log.debug("newOrigin: " + str(newOrigin_arr))
        self.kpoints = self.kpoints - newOrigin_arr
        self.log.debug("new Kpoints:\n" + str(self.kpoints))
        self.log.debug("Translate(): ...Done")
        return
