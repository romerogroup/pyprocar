import numpy as np

from pyprocar.utils.math import fft_interpolate_nd_3dmesh


def test_fft_interpolate_nd_3dmesh_refines_a_plain_3d_mesh_exactly():
    n = 8
    mesh = np.cos(2 * np.pi * np.arange(n) / n)[:, None, None] * np.ones((n, 4, 4))

    refined = fft_interpolate_nd_3dmesh(mesh, 2)

    assert refined.shape == (16, 8, 8)
    expected = np.cos(2 * np.pi * np.arange(16) / 16)
    np.testing.assert_allclose(refined[:, 3, 5], expected, atol=1e-12)
