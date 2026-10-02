"""Series building shared by the band, DOS, Fermi-surface and 2D-band plotters.

A plot is a list of series, one per (band, spin channel). Line plotters
(bands, DOS) lay their data out as ``(n_points, n_bands, n_channels)``;
surface plotters (Fermi, 2D bands) select each series' points with a mask.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

Lim = tuple[float, float]


class ShowColorbar(Enum):
    SINGLE = "single"
    PER_CHANNEL = "per_channel"
    NONE = "none"

    @classmethod
    def from_string(cls, input: str | ShowColorbar | bool | None) -> ShowColorbar:
        if isinstance(input, ShowColorbar):
            return input
        elif input is None:
            return cls.NONE
        elif isinstance(input, bool):
            return cls.SINGLE if input else cls.NONE

        string = input.lower()
        if string == "single":
            return cls.SINGLE
        elif string == "per_channel":
            return cls.PER_CHANNEL
        elif string == "none":
            return cls.NONE
        else:
            err_msg = f"Invalid colorbar mode: {string}. Valid modes are:\n"
            err_msg += "\n".join([f"- {mode}" for mode in cls.list_modes()])
            raise ValueError(err_msg)

    @classmethod
    def list_modes(cls) -> list[str]:
        """List all available colorbar modes."""
        return [mode.value for mode in cls]


class ChannelMode(Enum):
    FLIP = "flip"
    NORMAL = "normal"

    @classmethod
    def from_string(cls, string: str | ChannelMode | None) -> ChannelMode:
        if isinstance(string, ChannelMode):
            return string
        elif string is None:
            return cls.NORMAL

        string = string.lower()
        if string == "flip":
            return cls.FLIP
        elif string == "normal":
            return cls.NORMAL
        else:
            err_msg = f"Invalid channel mode: {string}. Valid modes are:\n"
            err_msg += "\n".join([f"- {mode}" for mode in cls.list_modes()])
            raise ValueError(err_msg)

    @classmethod
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]


@dataclass
class LineSeries:
    x: np.ndarray
    y: np.ndarray
    scalars: np.ndarray | None
    vectors: np.ndarray | None
    label: str | None
    band_index: int
    spin_index: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceSeries:
    mesh: pv.PolyData
    scalars: np.ndarray | None
    vectors: np.ndarray | None
    scalars_label: str | None
    band_index: int
    spin_index: int
    kwargs: dict[str, Any] = field(default_factory=dict)


def distribute_kwargs(kwargs: Mapping[str, Any], n_channels: int) -> list[dict[str, Any]]:
    """Split every list kwarg whose length equals ``n_channels`` across channels."""
    return [
        {
            key: value[i] if isinstance(value, list) and len(value) == n_channels else value
            for key, value in kwargs.items()
        }
        for i in range(n_channels)
    ]


def _check_layout(
    name: str, values: np.ndarray | None, y_shape: tuple[int, ...], share_single_channel: bool
) -> None:
    if values is None or values.ndim == 1:
        return
    _, n_bands, n_channels = y_shape
    bands_ok = values.shape[1] == n_bands
    channels_ok = values.ndim == 2 or values.shape[2] in (
        (n_channels, 1) if share_single_channel else (n_channels,)
    )
    if not (bands_ok and channels_ok):
        raise ValueError(f"{name} shape {values.shape} does not match the bands layout {y_shape}")


def _take(values: np.ndarray | None, iband: int, ichannel: int) -> np.ndarray | None:
    if values is None or values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[:, iband]
    return values[:, iband, ichannel if values.shape[2] > 1 else 0]


def line_series(
    x: np.ndarray,
    y: np.ndarray,
    scalars: np.ndarray | None,
    vectors: np.ndarray | None,
    channel_mode: ChannelMode | str,
    kwargs: Mapping[str, Any],
    labels: Callable[[int, int], str | None] | None = None,
    share_single_channel: bool = False,
) -> list[LineSeries]:
    """One series per (band, channel) of ``y``, shaped ``(n_points, n_bands, n_channels)``.

    ``scalars`` and ``vectors`` are sliced per series: 3D arrays by band and
    channel, 2D arrays by band, 1D arrays are shared by every series. Their band
    and channel counts must match ``y``, except that ``share_single_channel``
    lets a single channel colour every channel.
    """
    _check_layout("scalars", scalars, y.shape, share_single_channel)
    _check_layout("vectors", vectors, y.shape, share_single_channel)
    flip = ChannelMode.from_string(channel_mode) is ChannelMode.FLIP
    _, n_bands, n_channels = y.shape
    per_channel = distribute_kwargs(kwargs, n_channels)
    series = []
    for iband in range(n_bands):
        for ichannel in range(n_channels):
            y_series = y[:, iband, ichannel].copy()
            if flip and ichannel != 0:
                y_series *= -1.0
            series.append(
                LineSeries(
                    x=x,
                    y=y_series,
                    scalars=_take(scalars, iband, ichannel),
                    vectors=_take(vectors, iband, ichannel),
                    label=labels(iband, ichannel) if labels else None,
                    band_index=iband,
                    spin_index=ichannel,
                    kwargs=per_channel[ichannel],
                )
            )
    return series


def surface_series(
    surfaces: Mapping[tuple[int, int], pv.PolyData],
    masks: Mapping[tuple[int, int], np.ndarray],
    scalars_data,
    vectors_data,
    kwargs: Mapping[str, Any],
) -> list[SurfaceSeries]:
    """One series per (band, spin) surface, selecting each surface's points by mask."""
    scalars = scalars_data.to_array() if scalars_data is not None else None
    vectors = vectors_data.to_array() if vectors_data is not None else None
    return [
        SurfaceSeries(
            mesh=surface,
            scalars=scalars[masks[key]] if scalars is not None else None,
            vectors=vectors[masks[key]] if vectors is not None else None,
            scalars_label=scalars_data.label if scalars_data else None,
            band_index=key[0],
            spin_index=key[1],
            kwargs=dict(kwargs),
        )
        for key, surface in surfaces.items()
    ]


def channel_lims(data_lim, n_channels: int) -> list[Lim | None]:
    """Per-channel (min, max) from a Property's ``(n_channels, 2 * k)`` data limits.

    Each row holds k minima followed by k maxima (k is the band count for
    band data and 1 for DOS); the channel limit spans all of them.
    """
    lims: list[Lim | None] = [None] * n_channels
    arr = np.asarray(data_lim) if data_lim is not None else None
    if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
        return lims
    half = arr.shape[1] // 2
    for i in range(min(n_channels, arr.shape[0])):
        lims[i] = (float(np.min(arr[i, :half])), float(np.max(arr[i, half:])))
    return lims


def resolve_clim(
    user_clim: Lim | None,
    lims: Iterable[Lim | None],
    values: Iterable[np.ndarray | None],
) -> Lim:
    """The user's limits, else the union of known limits, else the finite data range."""
    if user_clim is not None:
        return user_clim
    known = [lim for lim in lims if lim is not None]
    if known:
        return (min(lim[0] for lim in known), max(lim[1] for lim in known))
    return finite_range(values)


def finite_range(values: Iterable[np.ndarray | None]) -> Lim:
    """(min, max) over the finite entries of every array, or (0, 1) if there are none."""
    arrays = [np.ravel(v) for v in values if v is not None]
    finite = np.concatenate(arrays) if arrays else np.empty(0)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    return (float(finite.min()), float(finite.max()))
