#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path
from typing import cast

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.colors import colorConverter  # type: ignore[attr-defined]

# Xticks type: [list[str] labels, list[int] positions]
XTicksType = list[list[str] | list[int]]


def plot_band_weight(
    kslist: list[npt.NDArray[np.float64]],
    ekslist: npt.NDArray[np.float64],
    wkslist: npt.NDArray[np.float64] | None = None,
    efermi: float | None = None,
    shift_efermi: bool = False,
    yrange: tuple[float, float] | None = None,
    output: str | None = None,
    style: str = "alpha",
    color: str = "blue",
    axis: Axes | None = None,
    width: int = 10,
    fatness: int = 4,
    xticks: XTicksType | None = None,
    cmap: Colormap = cm.bwr,
    weight_min: float = -0.1,
    weight_max: float = 0.6,
) -> Axes:
    del output  # unused parameter
    a: Axes
    if axis is None:
        _, ax = plt.subplots()
        a = cast(Axes, ax)
    else:
        a = axis
    ekslist_arr: npt.NDArray[np.float64]
    if efermi is not None and shift_efermi:
        ekslist_arr = np.array(ekslist) - efermi
    else:
        ekslist_arr = np.array(ekslist)

    if yrange is None:
        flat_arr: npt.NDArray[np.float64] = np.array(ekslist_arr).flatten()
        min_val = float(cast(np.float64, flat_arr.min()))
        max_val = float(cast(np.float64, flat_arr.max()))
        yrange = (
            min_val - 0.66,
            max_val + 0.66,
        )
    if wkslist is not None:
        for i in range(len(kslist)):
            x: npt.NDArray[np.float64] = kslist[i]
            y: npt.NDArray[np.float64] = cast(npt.NDArray[np.float64], ekslist_arr[i])
            # lwidths=np.ones(len(x))
            points: npt.NDArray[np.float64] = np.array([x, y]).T.reshape(-1, 1, 2)
            segments: list[npt.NDArray[np.float64]] = [points[:-1], points[1:]]
            segments_arr: npt.NDArray[np.float64] = np.concatenate(segments, axis=1)
            lc: LineCollection
            wks_row: npt.NDArray[np.float64] = cast(npt.NDArray[np.float64], wkslist[i])
            # Cast tolist() results since numpy stubs return Any
            segments_list = cast(list[list[list[float]]], segments_arr.tolist())
            if style == "width":
                lwidths: npt.NDArray[np.float64] = wks_row * width
                lwidths_list = cast(list[float], lwidths.tolist())
                lc = LineCollection(segments_list, linewidths=lwidths_list, colors=color)
            elif style == "alpha":
                lwidths = wks_row * width

                # The alpha values sometimes goes above 1 so in those cases we will normalize
                # the alpha values. -Uthpala
                lwidths_list_alpha = cast(list[float], lwidths.tolist())
                alpha_values: list[float] = [lw / (width + 0.05) for lw in lwidths_list_alpha]

                if max(alpha_values) > 1:
                    print("alpha is larger than 1. Renormalizing values.")
                    alpha_values = [alpha_i / max(alpha_values) for alpha_i in alpha_values]

                lc = LineCollection(
                    segments_list,
                    linewidths=[fatness] * len(x),
                    colors=[
                        colorConverter.to_rgba(color, alpha=alpha_i)
                        for alpha_i in alpha_values
                    ],
                )

            elif style == "color" or style == "colormap":
                lwidths = wks_row * width
                lwidths_list_color = cast(list[float], lwidths.tolist())
                norm = mpl.colors.Normalize(vmin=weight_min, vmax=weight_max)
                # norm = mpl.colors.SymLogNorm(linthresh=0.03,vmin=weight_min, vmax=weight_max)
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
                # lc = LineCollection(segments,linewidths=np.abs(norm(lwidths)-0.5)*1, colors=[m.to_rgba(lwidth) for lwidth in lwidths])
                lc = LineCollection(
                    segments_list,
                    linewidths=lwidths_list_color,
                    colors=[
                        cast(tuple[float, float, float, float], m.to_rgba(lw))
                        for lw in lwidths_list_color
                    ],
                )
            else:
                raise ValueError(f"Unknown style: {style}")
            _ = a.add_collection(lc)
    if axis is None:
        for i_band in range(len(kslist)):
            ks_band: npt.NDArray[np.float64] = kslist[i_band]
            eks_band: npt.NDArray[np.float64] = cast(npt.NDArray[np.float64], ekslist_arr[i_band])
            _ = a.plot(ks_band, eks_band, color="gray", linewidth=0.01)
        # a.set_xlim(0, xmax)
        # a.set_ylim(yrange)
        if xticks is not None:
            tick_positions: list[int] = cast(list[int], xticks[1])
            tick_labels: list[str] = cast(list[str], xticks[0])
            _ = a.set_xticks(tick_positions)
            _ = a.set_xticklabels(tick_labels)
            for xt in tick_positions:
                _ = a.axvline(xt, alpha=0.6, color="black", linewidth=0.7)
        if efermi is not None:
            if shift_efermi:
                _ = a.axhline(linestyle="--", color="black")
            else:
                _ = a.axhline(efermi, linestyle="--", color="black")

    return a


def main() -> None:
    parser = argparse.ArgumentParser(description="plot wannier bands.")
    _ = parser.add_argument("fname", type=str, help="dat filename")
    _ = parser.add_argument("-e", "--efermi", type=float, help="Fermi energy", default=None)
    _ = parser.add_argument("-o", "--output", type=str, help="output filename", default=None)
    _ = parser.add_argument("-w", "--weight", action="store_true", help="use -w to plot weighted band.")
    _ = parser.add_argument(
        "-y", "--yrange", type=float, nargs="+", help="range of yticks", default=None
    )
    _ = parser.add_argument(
        "-s", "--style", type=str, help="style of line, width | alpha", default="width"
    )
    args = parser.parse_args()
    output: str
    fname_arg: str = cast(str, args.fname)
    fname: str = fname_arg
    output_raw: str | None = cast(str | None, args.output)
    if output_raw is None:
        output = os.path.splitext(fname)[0] + ".png"
    else:
        output = output_raw

    # Note: This main function is incomplete - get_fermi and plot_band_weight_file
    # functions are not defined. This is legacy CLI code that needs the functions
    # to be imported or implemented.
    efermi_raw: float | None = cast(float | None, args.efermi)
    efermi: float | None = efermi_raw
    if efermi is not None:
        raise NotImplementedError(
            "This CLI main function is incomplete. "
            "get_fermi and plot_band_weight_file functions are not implemented."
        )

    plt.savefig(output)
    plt.show()


if __name__ == "__main__":
    main()
