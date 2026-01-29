"""Type stubs for pyvista.plotting.plotter.Plotter class."""

from _typeshed import Incomplete
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from pyvista.core.pointset import PolyData

_Self = TypeVar("_Self")

ColorLike = str | tuple[float, float, float] | tuple[float, float, float, float]

class Plotter:
    """PyVista Plotter class with complete type annotations for methods we use."""

    camera_position: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | str
    )
    off_screen: bool
    plane_widgets: list[Incomplete]

    def __init__(
        self,
        off_screen: bool | None = ...,
        notebook: bool | None = ...,
        shape: tuple[int, int] | str = ...,
        groups: Incomplete = ...,
        row_weights: Incomplete = ...,
        col_weights: Incomplete = ...,
        border: bool = ...,
        border_color: ColorLike = ...,
        border_width: float = ...,
        window_size: tuple[int, int] | None = ...,
        multi_samples: int | None = ...,
        line_smoothing: bool = ...,
        polygon_smoothing: bool = ...,
        splitting_position: float | None = ...,
        title: str | None = ...,
        lighting: str | None = ...,
        theme: Incomplete = ...,
        image_scale: int = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def add_mesh(
        self,
        mesh: PolyData | Incomplete,
        color: ColorLike | None = ...,
        style: str | None = ...,
        scalars: str | npt.NDArray[np.float64] | None = ...,
        clim: tuple[float, float] | None = ...,
        show_edges: bool | None = ...,
        edge_color: ColorLike | None = ...,
        point_size: float = ...,
        line_width: float | None = ...,
        opacity: float = ...,
        flip_scalars: bool = ...,
        lighting: bool | None = ...,
        n_colors: int = ...,
        interpolate_before_map: bool = ...,
        cmap: str | Incomplete | None = ...,
        label: str | None = ...,
        reset_camera: bool | None = ...,
        scalar_bar_args: dict[str, Incomplete] | None = ...,
        show_scalar_bar: bool | None = ...,
        multi_colors: bool = ...,
        name: str | None = ...,
        texture: Incomplete = ...,
        render_points_as_spheres: bool | None = ...,
        render_lines_as_tubes: bool = ...,
        smooth_shading: bool | None = ...,
        ambient: float = ...,
        diffuse: float = ...,
        specular: float = ...,
        specular_power: float = ...,
        nan_color: ColorLike | None = ...,
        nan_opacity: float = ...,
        culling: str | bool | None = ...,
        rgb: bool | None = ...,
        categories: bool = ...,
        silhouette: bool | dict[str, Incomplete] = ...,
        use_transparency: bool = ...,
        below_color: ColorLike | None = ...,
        above_color: ColorLike | None = ...,
        annotations: dict[float, str] | None = ...,
        pickable: bool = ...,
        preference: str = ...,
        log_scale: bool = ...,
        pbr: bool | None = ...,
        metallic: float | None = ...,
        roughness: float | None = ...,
        render: bool = ...,
        component: int | None = ...,
        emissive: bool = ...,
        copy_mesh: bool = ...,
        backface_params: dict[str, Incomplete] | None = ...,
        show_vertices: bool | None = ...,
        edge_opacity: float | None = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_point_labels(
        self,
        points: npt.NDArray[np.float64],
        labels: list[str | int] | str,
        italic: bool = ...,
        bold: bool = ...,
        font_size: int | None = ...,
        text_color: ColorLike | None = ...,
        font_family: str | None = ...,
        shadow: bool = ...,
        show_points: bool = ...,
        point_color: ColorLike | None = ...,
        point_size: float | None = ...,
        name: str | None = ...,
        shape_color: ColorLike | None = ...,
        shape: str | None = ...,
        shape_opacity: float = ...,
        fill_shape: bool = ...,
        margin: int = ...,
        tolerance: float = ...,
        reset_camera: bool | None = ...,
        always_visible: bool = ...,
        render_points_as_spheres: bool = ...,
        render: bool = ...,
        pickable: bool = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_text(
        self,
        text: str,
        position: str | tuple[float, float] = ...,
        font_size: int = ...,
        color: ColorLike | None = ...,
        font: str | None = ...,
        shadow: bool = ...,
        name: str | None = ...,
        viewport: bool = ...,
        orientation: float = ...,
        font_file: str | None = ...,
        render: bool = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def show(
        self,
        title: str | None = ...,
        window_size: tuple[int, int] | None = ...,
        interactive: bool = ...,
        auto_close: bool | None = ...,
        interactive_update: bool = ...,
        full_screen: bool | None = ...,
        screenshot: str | bool | None = ...,
        return_img: bool = ...,
        cpos: Incomplete = ...,
        jupyter_backend: str | None = ...,
        return_viewer: bool = ...,
        return_cpos: bool = ...,
        before_close_callback: Callable[..., Incomplete] | None = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def screenshot(
        self,
        filename: str | bool | None = ...,
        transparent_background: bool | None = ...,
        return_img: bool | None = ...,
        window_size: tuple[int, int] | None = ...,
        scale: int | None = ...,
    ) -> npt.NDArray[np.uint8] | None: ...
    def save_graphic(
        self,
        filename: str,
        title: str = ...,
        raster: bool = ...,
        painter: bool = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def add_slider_widget(
        self,
        callback: Callable[[float], Incomplete],
        rng: tuple[float, float] | list[float],
        value: float | None = ...,
        title: str | None = ...,
        pointa: tuple[float, float] = ...,
        pointb: tuple[float, float] = ...,
        color: ColorLike | None = ...,
        pass_widget: bool = ...,
        interaction_event: str | int = ...,
        style: str | None = ...,
        title_height: float | None = ...,
        title_opacity: float = ...,
        title_color: ColorLike | None = ...,
        fmt: str | None = ...,
        slider_width: float | None = ...,
        tube_width: float | None = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_plane_widget(
        self,
        callback: Callable[
            [tuple[float, float, float], tuple[float, float, float]], Incomplete
        ],
        normal: str | tuple[float, float, float] = ...,
        origin: tuple[float, float, float] | None = ...,
        bounds: tuple[float, float, float, float, float, float] | None = ...,
        factor: float = ...,
        resolution: int = ...,
        color: ColorLike | None = ...,
        assign_to_axis: str | int | None = ...,
        tubing: bool = ...,
        outline_translation: bool = ...,
        origin_translation: bool = ...,
        implicit: bool = ...,
        pass_widget: bool = ...,
        test_callback: bool = ...,
        normal_rotation: bool = ...,
        interaction_event: str | int = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_box_widget(
        self,
        callback: Callable[[Incomplete], Incomplete],
        bounds: tuple[float, float, float, float, float, float] | None = ...,
        factor: float = ...,
        rotation_enabled: bool = ...,
        color: ColorLike | None = ...,
        use_planes: bool = ...,
        outline_translation: bool = ...,
        pass_widget: bool = ...,
        interaction_event: str | int = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_scalar_bar(
        self,
        title: str = ...,
        n_labels: int = ...,
        italic: bool = ...,
        bold: bool = ...,
        title_font_size: int | None = ...,
        label_font_size: int | None = ...,
        color: ColorLike | None = ...,
        font_family: str | None = ...,
        shadow: bool = ...,
        width: float | None = ...,
        height: float | None = ...,
        position_x: float | None = ...,
        position_y: float | None = ...,
        vertical: bool | None = ...,
        interactive: bool | None = ...,
        fmt: str | None = ...,
        use_opacity: bool = ...,
        outline: bool = ...,
        nan_annotation: bool = ...,
        below_label: str | None = ...,
        above_label: str | None = ...,
        background_color: ColorLike | None = ...,
        n_colors: int | None = ...,
        fill: bool = ...,
        render: bool = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def show_grid(
        self,
        color: ColorLike | None = ...,
        show_xlabels: bool = ...,
        show_ylabels: bool = ...,
        show_zlabels: bool = ...,
        xlabel: str = ...,
        ylabel: str = ...,
        zlabel: str = ...,
        xtitle: str = ...,
        ytitle: str = ...,
        ztitle: str = ...,
        font_size: int | None = ...,
        font_family: str | None = ...,
        bold: bool = ...,
        all_edges: bool = ...,
        corner_factor: float = ...,
        fmt: str | None = ...,
        minor_ticks: bool = ...,
        padding: float = ...,
        use_2d: bool = ...,
        grid: str | bool = ...,
        location: str = ...,
        ticks: str = ...,
        render: bool = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def add_axes(
        self,
        interactive: bool | None = ...,
        line_width: int = ...,
        color: ColorLike | None = ...,
        x_color: ColorLike | None = ...,
        y_color: ColorLike | None = ...,
        z_color: ColorLike | None = ...,
        xlabel: str = ...,
        ylabel: str = ...,
        zlabel: str = ...,
        labels_off: bool = ...,
        box: bool | None = ...,
        box_args: dict[str, Incomplete] | None = ...,
        viewport: tuple[float, float, float, float] = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def set_background(
        self,
        color: ColorLike = ...,
        top: ColorLike | None = ...,
        all_renderers: bool = ...,
    ) -> None: ...
    def orbit_on_path(
        self,
        path: Incomplete = ...,
        focus: Incomplete = ...,
        step: float = ...,
        viewup: Incomplete = ...,
        write_frames: bool = ...,
        threaded: bool = ...,
        progress_bar: bool = ...,
    ) -> Incomplete: ...
    def open_movie(
        self,
        filename: str,
        framerate: int = ...,
        quality: int = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def close(self) -> None: ...
    def remove_scalar_bar(self) -> None: ...
    def view_isometric(self) -> None: ...
    def view_xy(self, negative: bool = ...) -> None: ...
    def view_xz(self, negative: bool = ...) -> None: ...
    def view_yz(self, negative: bool = ...) -> None: ...
    def open_gif(self, filename: str, loop: int = ..., fps: int = ...) -> None: ...
    def write_frame(self) -> None: ...
    def add_legend(
        self,
        labels: list[tuple[str, ColorLike]] | None = ...,
        bcolor: ColorLike | None = ...,
        border: bool = ...,
        size: tuple[float, float] = ...,
        name: str | None = ...,
        loc: str = ...,
        face: str | PolyData | None = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
