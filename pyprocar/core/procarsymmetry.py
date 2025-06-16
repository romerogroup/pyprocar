import sys
import re
import logging

import numpy as np
import matplotlib.pyplot as plt

class ProcarSymmetry:
    def __init__(
        self,
        kpoints,
        bands,
        character=None,
        sx=None,
        sy=None,
        sz=None,
        loglevel=logging.WARNING,
    ):
        self.log = logging.getLogger("ProcarSymmetry")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(
            logging.Formatter("%(name)s::%(levelname)s: " "%(message)s")
        )
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        self.log.debug("ProcarSymmetry.__init__: ...")

        self.kpoints = kpoints
        self.bands = bands
        # optional arguments when not given will False, but they can still
        # be treated like arrays
        self.character = np.array([])
        if character is not None:
            self.character = character
        self.sx = np.array([])
        if sx is not None:
            self.sx = sx
        self.sy = np.array([])
        if sy is not None:
            self.sy = sy
        self.sz = np.array([])
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

    def _q_mult(self, q1, q2):
        """
    Multiplication of quaternions, it doesn't fit in any other place
    """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array((w, x, y, z))

    def general_rotation(self, angle, rotAxis=[0, 0, 1], store=True):
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
        if rotAxis == "x" or rotAxis == "X":
            rotAxis = [1, 0, 0]
        if rotAxis == "y" or rotAxis == "Y":
            rotAxis = [0, 1, 0]
        if rotAxis == "z" or rotAxis == "Z":
            rotAxis = [0, 0, 1]
        rotAxis = np.array(rotAxis, dtype=float)
        self.log.debug("rotAxis : " + str(rotAxis))
        rotAxis = rotAxis / np.linalg.norm(rotAxis)
        self.log.debug("rotAxis Normalized : " + str(rotAxis))
        self.log.debug("Angle : " + str(angle))
        angle = angle * np.pi / 180
        # defining a quaternion for rotatoin
        angle = angle / 2
        rotAxis = rotAxis * np.sin(angle)
        qRot = np.array((np.cos(angle), rotAxis[0], rotAxis[1], rotAxis[2]))
        qRotI = np.array((np.cos(angle), -rotAxis[0], -rotAxis[1], -rotAxis[2]))
        self.log.debug("Rot. quaternion : " + str(qRot))
        self.log.debug("Rot. quaternion conjugate : " + str(qRotI))
        # converting self.kpoints into quaternions
        w = np.zeros((len(self.kpoints), 1))
        qvectors = np.column_stack((w, self.kpoints)).transpose()
        self.log.debug(
            "Kpoints-> quaternions (transposed):\n" + str(qvectors.transpose())
        )
        qvectors = self._q_mult(qRot, qvectors)
        qvectors = self._q_mult(qvectors, qRotI).transpose()
        kpoints = qvectors[:, 1:]
        self.log.debug("Rotated kpoints :\n" + str(qvectors))

        # rotating the spin vector (if exist)
        sxShape, syShape, szShape = self.sx.shape, self.sy.shape, self.sz.shape
        self.log.debug("Spin vector Shapes : " + str((sxShape, syShape, szShape)))
        # The first entry has to be an array of 0s, w could do the work,
        # but if len(self.sx)==0 qvectors will have a non-defined length
        qvectors = (
            0 * self.sx.flatten(),
            self.sx.flatten(),
            self.sy.flatten(),
            self.sz.flatten(),
        )
        self.log.debug("Spin vector quaternions: \n" + str(qvectors))
        qvectors = self._q_mult(qRot, qvectors)
        qvectors = self._q_mult(qvectors, qRotI)
        self.log.debug("Spin quaternions after rotation:\n" + str(qvectors))
        sx, sy, sz = qvectors[1], qvectors[2], qvectors[3]
        sx.shape, sy.shape, sz.shape = sxShape, syShape, szShape

        if store is True:
            self.kpoints, self.sx, self.sy, self.sz = kpoints, sx, sy, sz
        self.log.debug("GeneralRotation: ...Done")
        return (kpoints, sx, sy, sz)

    def rot_symmetry_z(self, order):
        """Applies the given rotational crystal symmetry to the current
    system. ie: to unfold the irreductible BZ to the full BZ.

    Only rotations along z-axis are performed, you can use
    self.GeneralRotation first. 

    The user is responsible of provide a useful input. The method
    doesn't check the physics.

    """
        self.log.debug("RotSymmetryZ:...")
        rotations = [
            self.general_rotation(360 * i / order, store=False) for i in range(order)
        ]
        rotations = list(zip(*rotations))
        self.log.debug(
            "self.kpoints.shape (before concat.): " + str(self.kpoints.shape)
        )
        self.kpoints = np.concatenate(rotations[0], axis=0)
        self.log.debug("self.kpoints.shape (after concat.): " + str(self.kpoints.shape))
        self.sx = np.concatenate(rotations[1], axis=0)
        self.sy = np.concatenate(rotations[2], axis=0)
        self.sz = np.concatenate(rotations[3], axis=0)
        # the bands and proj. character also need to be enlarged
        bandsChar = [(self.bands, self.character) for i in range(order)]
        bandsChar = list(zip(*bandsChar))
        self.bands = np.concatenate(bandsChar[0], axis=0)
        self.character = np.concatenate(bandsChar[1], axis=0)
        self.log.debug("RotSymmZ:...Done")

        return

    def mirror_x(self):
        """Applies the given rotational crystal symmetry to the current
    system. ie: to unfold the irreductible BZ to the full BZ.

    """
        self.log.debug("Mirror:...")
        newK = self.kpoints * np.array([1, -1, 1])
        self.kpoints = np.concatenate((self.kpoints, newK), axis=0)
        self.log.debug("self.kpoints.shape (after concat.): " + str(self.kpoints.shape))
        newSx = -1 * self.sx
        newSy = 1 * self.sy
        newSz = 1 * self.sz
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

    def translate(self, newOrigin):
        """Centers the Kpoints at newOrigin, newOrigin is either and index (of
   some Kpoint) or the cartesian coordinates of one point in the
   reciprocal space.

    """
        self.log.debug("Translate():  ...")
        if len(newOrigin) == 1:
            newOrigin = int(newOrigin[0])
            newOrigin = self.kpoints[newOrigin]
        # Make sure newOrigin is a numpy array
        newOrigin = np.array(newOrigin, dtype=float)
        self.log.debug("newOrigin: " + str(newOrigin))
        self.kpoints = self.kpoints - newOrigin
        self.log.debug("new Kpoints:\n" + str(self.kpoints))
        self.log.debug("Translate(): ...Done")
        return
