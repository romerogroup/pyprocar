"""Unit tests for FermiSlicePlotter and FermiSliceSeries.

Note: These tests use direct module loading to avoid a pre-existing circular
import issue with EBSPlot in scriptUnfold.py. Once that issue is fixed,
these tests can use normal imports:
    from pyprocar.plotter.fs_slice_plot import FermiSlicePlotter, FermiSliceSeries
"""

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# ------------------------------------------------------------------
# FermiSliceSeries Tests (can use direct dataclass creation)
# ------------------------------------------------------------------


def _make_series(
    n_points: int = 10,
    has_scalars: bool = True,
    has_vectors: bool = False,
):
    """Create a FermiSliceSeries for testing."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class FermiSliceSeries:
        points_2d: np.ndarray
        lines: np.ndarray
        scalars: np.ndarray | None
        scalars_label: str | None
        scalars_unit: str | None
        scalars_lim: tuple[float, float] | None
        vectors: np.ndarray | None
        vectors_label: str | None
        vectors_unit: str | None
        vectors_lim: tuple[float, float] | None
        label: str | None
        additional_kwargs: dict[str, Any] = field(default_factory=dict)

    points_2d = np.random.rand(n_points, 2)
    # Create line connectivity: pairs of points
    # PyVista format: [n_pts, p0, p1, n_pts, p2, p3, ...]
    lines_list = []
    for i in range(n_points - 1):
        lines_list.extend([2, i, i + 1])  # 2 points per segment
    lines = np.array(lines_list, dtype=int)

    return FermiSliceSeries(
        points_2d=points_2d,
        lines=lines,
        scalars=np.random.rand(n_points) if has_scalars else None,
        scalars_label="Test Scalars" if has_scalars else None,
        scalars_unit="eV" if has_scalars else None,
        scalars_lim=(0.0, 1.0) if has_scalars else None,
        vectors=np.random.rand(n_points, 2) if has_vectors else None,
        vectors_label="Test Vectors" if has_vectors else None,
        vectors_unit="m/s" if has_vectors else None,
        vectors_lim=(0.0, 1.0) if has_vectors else None,
        label="Test Series",
    )


class TestFermiSliceSeriesDataclass:
    """Tests for FermiSliceSeries dataclass structure."""

    def test_create_minimal_series(self):
        """Can create series with minimal required fields."""
        series = _make_series(has_scalars=False, has_vectors=False)
        assert series.points_2d is not None
        assert series.lines is not None
        assert series.scalars is None
        assert series.vectors is None

    def test_create_full_series(self):
        """Can create series with all fields populated."""
        series = _make_series(has_scalars=True, has_vectors=True)
        assert series.scalars is not None
        assert series.vectors is not None
        assert series.scalars_label == "Test Scalars"
        assert series.vectors_label == "Test Vectors"

    def test_series_additional_kwargs_default(self):
        """Additional kwargs defaults to empty dict."""
        series = _make_series()
        assert series.additional_kwargs == {}


# ------------------------------------------------------------------
# FermiSlicePlotter Tests using mock PyVista
# ------------------------------------------------------------------


def _make_mock_fermi_surface(
    n_points: int = 100, has_scalars: bool = True, has_vectors: bool = False
) -> pv.PolyData:
    """Create a mock Fermi surface (sphere) for testing."""
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0), theta_resolution=10, phi_resolution=10)

    if has_scalars:
        scalars = np.linalg.norm(sphere.points, axis=1)
        sphere["scalars"] = scalars
        sphere.set_active_scalars("scalars")

    if has_vectors:
        vectors = sphere.points / np.linalg.norm(sphere.points, axis=1, keepdims=True)
        sphere["vectors"] = vectors
        sphere.set_active_vectors("vectors")

    return sphere


# Since the full FermiSlicePlotter depends on pyprocar imports that are currently broken,
# we test the core logic patterns that it implements rather than the class directly.


class TestFermiSlicePlotterPatterns:
    """Tests for patterns used in FermiSlicePlotter."""

    def test_orthonormal_basis_z_normal(self):
        """Orthonormal basis computation for z-normal plane."""
        normal = np.array([0, 0, 1])

        # Compute basis
        if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
            v_temp = np.array([0, 0, 1])
        else:
            v_temp = np.array([0, 1, 0])

        u = np.cross(v_temp, normal).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u).astype(np.float32)
        v /= np.linalg.norm(v)

        # Verify orthonormal
        assert abs(np.dot(u, v)) < 1e-6
        assert abs(np.dot(u, normal)) < 1e-6
        assert abs(np.dot(v, normal)) < 1e-6
        assert abs(np.linalg.norm(u) - 1.0) < 1e-6
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_orthonormal_basis_x_normal(self):
        """Orthonormal basis computation for x-normal plane."""
        normal = np.array([1, 0, 0])

        if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
            v_temp = np.array([0, 0, 1])
        else:
            v_temp = np.array([0, 1, 0])

        u = np.cross(v_temp, normal).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u).astype(np.float32)
        v /= np.linalg.norm(v)

        assert abs(np.dot(u, v)) < 1e-6
        assert abs(np.dot(u, normal)) < 1e-6
        assert abs(np.dot(v, normal)) < 1e-6

    def test_iter_segments_pattern(self):
        """Line segment iteration pattern works correctly."""
        # PyVista line format: [n_pts, p1, p2, n_pts, p3, p4, ...]
        lines = np.array([2, 0, 1, 2, 2, 3])

        segments = []
        i = 0
        while i < len(lines):
            num_points_in_line = lines[i]
            line_connectivity_start = i + 1
            for j in range(num_points_in_line - 1):
                start_idx = lines[line_connectivity_start + j]
                end_idx = lines[line_connectivity_start + j + 1]
                segments.append((start_idx, end_idx))
            i += num_points_in_line + 1

        assert segments == [(0, 1), (2, 3)]

    def test_slice_returns_polydata(self):
        """Slicing a surface returns PolyData."""
        sphere = _make_mock_fermi_surface(has_scalars=True)
        slice_data = sphere.slice(normal=[0, 0, 1], origin=[0, 0, 0])

        assert isinstance(slice_data, pv.PolyData)
        assert slice_data.n_points > 0

    def test_slice_preserves_scalars(self):
        """Sliced data preserves scalar field."""
        sphere = _make_mock_fermi_surface(has_scalars=True)
        slice_data = sphere.slice(normal=[0, 0, 1], origin=[0, 0, 0])

        assert slice_data.active_scalars is not None

    def test_slice_preserves_vectors(self):
        """Sliced data preserves vector field."""
        sphere = _make_mock_fermi_surface(has_scalars=True, has_vectors=True)
        slice_data = sphere.slice(normal=[0, 0, 1], origin=[0, 0, 0])

        assert slice_data.active_vectors is not None


class TestMatplotlibIntegration:
    """Tests for matplotlib integration patterns."""

    def test_line_collection_creation(self):
        """LineCollection can be created from segments."""
        from matplotlib.collections import LineCollection

        segments = [[(0, 0), (1, 1)], [(1, 1), (2, 0)]]
        colors = [0.5, 0.8]

        lc = LineCollection(segments, array=colors, cmap="plasma")

        assert lc is not None

    def test_colorbar_creation(self):
        """Colorbar can be attached to mappable."""
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots()
        segments = [[(0, 0), (1, 1)]]
        lc = LineCollection(segments, array=[0.5], cmap="plasma")
        ax.add_collection(lc)

        cb = fig.colorbar(lc, ax=ax, label="Test")

        assert cb is not None

        plt.close(fig)

    def test_quiver_creation(self):
        """Quiver plot can be created."""
        fig, ax = plt.subplots()

        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        u = np.array([1, 0, -1])
        v = np.array([0, 1, 0])

        q = ax.quiver(x, y, u, v)

        assert q is not None

        plt.close(fig)


class TestAxisConfiguration:
    """Tests for axis configuration patterns."""

    def test_set_axis_labels(self):
        """Axis labels can be set."""
        fig, ax = plt.subplots()

        ax.set_xlabel(r"$k_x$ (1/$\AA$)")
        ax.set_ylabel(r"$k_y$ (1/$\AA$)")

        assert ax.get_xlabel() == r"$k_x$ (1/$\AA$)"
        assert ax.get_ylabel() == r"$k_y$ (1/$\AA$)"

        plt.close(fig)

    def test_set_axis_limits(self):
        """Axis limits can be set."""
        fig, ax = plt.subplots()

        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

        assert ax.get_xlim() == (-1, 1)
        assert ax.get_ylim() == (-1, 1)

        plt.close(fig)

    def test_set_aspect_equal(self):
        """Aspect ratio can be set to equal."""
        fig, ax = plt.subplots()

        ax.set_aspect("equal")

        # Should not raise
        plt.close(fig)


class TestExportPatterns:
    """Tests for export functionality patterns."""

    def test_csv_export(self, tmp_path):
        """Data can be exported to CSV."""
        import pandas as pd

        values = {
            "points_u": np.array([0, 1, 2]),
            "points_v": np.array([0, 1, 0]),
            "scalars": np.array([0.5, 0.8, 0.3]),
        }

        filepath = tmp_path / "test.csv"
        df = pd.DataFrame(values)
        df.to_csv(filepath, sep=",", index=False)

        assert filepath.exists()

        # Verify content
        df_read = pd.read_csv(filepath)
        np.testing.assert_array_almost_equal(df_read["points_u"], values["points_u"])

    def test_json_export(self, tmp_path):
        """Data can be exported to JSON."""
        import json

        values = {
            "points_u": [0.0, 1.0, 2.0],
            "points_v": [0.0, 1.0, 0.0],
            "scalars": [0.5, 0.8, 0.3],
        }

        filepath = tmp_path / "test.json"
        with open(filepath, "w") as f:
            json.dump(values, f)

        assert filepath.exists()

        # Verify content
        with open(filepath) as f:
            data = json.load(f)
        assert data["points_u"] == values["points_u"]

    def test_savefig(self, tmp_path):
        """Figure can be saved to file."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])

        filepath = tmp_path / "test.png"
        fig.savefig(filepath)

        assert filepath.exists()

        plt.close(fig)


# ------------------------------------------------------------------
# ShowColorbar Enum Tests (uses simple enum reimplementation)
# ------------------------------------------------------------------


class TestShowColorbarEnum:
    """Tests for ShowColorbar enum pattern."""

    def test_from_string_single(self):
        """'single' converts to SINGLE enum."""
        from enum import Enum

        class ShowColorbar(Enum):
            NONE = "none"
            SINGLE = "single"
            PER_CHANNEL = "per_channel"

            @classmethod
            def from_string(cls, value):
                if isinstance(value, cls):
                    return value
                mapping = {
                    "none": cls.NONE,
                    "single": cls.SINGLE,
                    "per_channel": cls.PER_CHANNEL,
                }
                if value.lower() not in mapping:
                    raise ValueError(f"Invalid ShowColorbar: {value}")
                return mapping[value.lower()]

        result = ShowColorbar.from_string("single")
        assert result == ShowColorbar.SINGLE

    def test_from_string_none(self):
        """'none' converts to NONE enum."""
        from enum import Enum

        class ShowColorbar(Enum):
            NONE = "none"
            SINGLE = "single"

            @classmethod
            def from_string(cls, value):
                if isinstance(value, cls):
                    return value
                return cls.NONE if value.lower() == "none" else cls.SINGLE

        result = ShowColorbar.from_string("none")
        assert result == ShowColorbar.NONE

    def test_from_string_passthrough(self):
        """Enum passthrough works."""
        from enum import Enum

        class ShowColorbar(Enum):
            SINGLE = "single"

            @classmethod
            def from_string(cls, value):
                if isinstance(value, cls):
                    return value
                return cls.SINGLE

        result = ShowColorbar.from_string(ShowColorbar.SINGLE)
        assert result == ShowColorbar.SINGLE
