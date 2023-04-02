import numpy as np
import pytest
from pyprocar.core.brillouin_zone import Lines, BrillouinZone

def test_Lines():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    lines = Lines(verts, faces)
    
    assert lines.nface == 4
    assert np.array_equal(lines.verts, verts)
    assert np.array_equal(lines.faces, faces)
    

def test_BrillouinZone():
    reciprocal_lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    brillouin_zone = BrillouinZone(reciprocal_lattice)

    # Check if verts and faces have been initialized
    assert hasattr(brillouin_zone, 'verts')
    assert hasattr(brillouin_zone, 'faces')

    # Check if the normals are properly fixed
    center = brillouin_zone.centers[0]
    n1 = center / np.linalg.norm(center)
    n2 = brillouin_zone.face_normals[0]
    assert np.dot(n1, n2) >= 0