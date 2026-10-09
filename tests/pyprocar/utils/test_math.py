import numpy as np
import pytest

from pyprocar.utils.math import fft_interpolate_mesh, fft_interpolate_nd_3dmesh


def test_fft_interpolate_nd_3dmesh_refines_a_plain_3d_mesh_exactly():
    n = 8
    mesh = np.cos(2 * np.pi * np.arange(n) / n)[:, None, None] * np.ones((n, 4, 4))

    refined = fft_interpolate_nd_3dmesh(mesh, 2)

    assert refined.shape == (16, 8, 8)
    expected = np.cos(2 * np.pi * np.arange(16) / 16)
    np.testing.assert_allclose(refined[:, 3, 5], expected, atol=1e-12)


@pytest.mark.parametrize("dtype", [float, complex])
def test_fft_interpolate_mesh_of_a_nyquist_term_is_real_cosine(dtype):
    # [1, -1, 1, -1] is cos(pi j) and all Nyquist; on the doubled grid it is cos(pi j / 2).
    samples = np.array([1, -1, 1, -1], dtype=dtype).reshape(4, 1, 1)

    refined = fft_interpolate_mesh(samples, 2)

    expected = np.array([1, 0, -1, 0, 1, 0, -1, 0]).reshape(8, 1, 1)
    assert refined.dtype == np.dtype(dtype)
    np.testing.assert_allclose(refined, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape", [(4, 1, 1), (6, 4, 2), (5, 3, 4), (8, 6, 4)])
def test_fft_interpolate_mesh_of_complex_values_is_the_interpolation_of_their_parts(shape):
    rng = np.random.default_rng(0)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    refined = fft_interpolate_mesh(values, 2)

    parts = fft_interpolate_mesh(values.real, 2) + 1j * fft_interpolate_mesh(values.imag, 2)
    np.testing.assert_allclose(refined, parts, rtol=0, atol=1e-12)
    # Every second point of an axis with more than one sample is an original sample.
    kept = tuple(slice(None, None, 2 if n > 1 else 1) for n in shape)
    np.testing.assert_allclose(refined[kept], values, rtol=0, atol=1e-12)
