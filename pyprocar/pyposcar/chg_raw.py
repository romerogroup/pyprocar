#!/usr/bin/env python3
from __future__ import annotations

######################
## TODO:
## -Chg.shift
## -Chg.plot_atoms (maybe in poscar.py)
## -Chg.Charge_redistributions
######################
import argparse
import math
import os
import re
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from . import plot3d
from .poscar import Poscar


class Chg_base:
    comment: str | None
    Ispin: int
    ngf: npt.NDArray[np.signedinteger[Any]]
    poscar: Poscar | None
    file_content: str
    Data_blocks: list[object]
    Data0: npt.NDArray[np.float64]
    Data1: npt.NDArray[np.float64]
    Data2: npt.NDArray[np.float64]
    Data3: npt.NDArray[np.float64]
    is_chg: bool
    is_locpot: bool | None
    verbose: bool | str

    def __init__(self) -> None:
        self.comment = None
        self.Ispin = 1
        self.ngf = np.array([], dtype=int)
        self.poscar = None
        self.file_content = ""
        self.Data_blocks = []
        self.Data0 = np.array([], dtype=np.float64)
        self.Data1 = np.array([], dtype=np.float64)
        self.Data2 = np.array([], dtype=np.float64)
        self.Data3 = np.array([], dtype=np.float64)
        self.is_chg = True
        self.is_locpot = None
        self.verbose = False

    def Load(
        self,
        filename: str = "CHGCAR",
        frame: int = 0,
        is_chg: bool | None = None,
        verbose: bool | str | None = None,
    ) -> None:
        """Load a CHG-like file

        `verbose` = False: No verbosity
                  = True: verbose output
                  = 'debug': usually unwanted verbosity level

        """
        if verbose is not None:
            self.verbose = verbose
        if is_chg is not None:
            self.is_chg = is_chg
        if self.verbose:
            print("\nINFO: Loading a CHG-like with the following parameters:")
            print("INFO: Filename: ", filename)
            print("INFO: is_chg:   ", self.is_chg)
        # checking if the file exist
        if not os.path.isfile(filename):
            print("ERROR: can't open the file, please check:", filename)
            raise RuntimeError("File does not exist")
        self.file_content = open(filename).read()

        # getting how many ionic steps were loaded
        comment = re.findall(r"^[^\n]*\n", self.file_content)[0]
        self.comment = comment
        if self.verbose == "debug":
            print("DEBUG: Comment line:")
            print(self.comment)

        Nframes = len(re.findall(comment, self.file_content))
        if self.verbose == "debug":
            print("DEBUG: Number of frames found:", Nframes)

        if self.verbose:
            print("INFO: Selecting the frame:", frame)
        frames: list[str] = re.split(comment, self.file_content)
        # if the first frame is empty (a `re` thing), should be discarded
        if len(frames[0]) == 0:
            frames.pop(0)
        self.file_content = frames[frame]

        # next string to find is the line with
        # NGFx NGFy NGFz
        ngf_line_list = re.findall(r"\n\s*\n(\s+\d+\s+\d+\s+\d+)\n", self.file_content)
        if verbose:
            print("INFO: NGFx NGFy NGFz", ngf_line_list)
        if len(ngf_line_list) != 1:
            raise RuntimeError("NGFline should have one and only one occurrence")
        ngf_line = ngf_line_list[0]
        self.ngf = np.array(ngf_line.split(), dtype=int)
        if verbose == "debug":
            print("DEBUG: grid size,", self.ngf)

        data: list[str] = re.split(ngf_line, self.file_content)
        ndata = len(data)
        if self.verbose:
            print("INFO: number of data blocks", ndata)

        # The first block is a POSCAR-like string
        # The second block is the spin-up
        # The third block (if present) is the spin-down
        # The 4th and 5th blocks are Sy, Sz (1s:rho, 2nd:Sx)
        if ndata == 2:
            self.Ispin = 1
            print("\nA non-magnetic calculation detected\n")
        elif ndata == 3:
            self.Ispin = 2
            print("\nA spin-polarized calculation detected\n")
        elif ndata == 5:
            self.Ispin = 4
            print("\nA non-collinear calculation detected\n")
        else:
            raise RuntimeError("Number of block data is unexpected," + str(ndata))

        # The poscar-like string would be passed to a POSCAR class
        assert self.comment is not None
        poscarString = self.comment + data.pop(0)
        self.poscar = Poscar(filename="")
        self.poscar.parse(fromString=poscarString)
        if self.verbose == "debug":
            print("DEBUG: POSCAR-like info:")
            assert self.poscar.poscar is not None
            if isinstance(self.poscar.poscar, list):
                print("\n".join(self.poscar.poscar))
            else:
                print(self.poscar.poscar)

        # Now we will search for augmentation charges
        # 'augmentation occupancies'
        # That info will be discarded
        cleaned: list[str] = []
        for block in data:
            cleaned.append(re.split(r"augmentation occupancies", block)[0])
        data = cleaned

        # The grid data will be processed
        Ndata = int(self.ngf[0]) * int(self.ngf[1]) * int(self.ngf[2])
        if self.verbose == "debug":
            print("DEBUG: Data points expected:", Ndata)

        data0_split = data[0].split()
        ngfx, ngfy, ngfz = int(self.ngf[0]), int(self.ngf[1]), int(self.ngf[2])

        # If the grid points don't agree it could be a LOCPOT with residual data
        if len(data0_split) != Ndata:
            assert self.poscar.Ntotal is not None
            N = self.poscar.Ntotal
            if len(data0_split) == Ndata + N:
                self.is_locpot = True
                self.Data0 = np.array(data0_split[:-N], dtype=np.float64).reshape(
                    ngfz, ngfy, ngfx
                )
                if self.verbose == "debug":
                    print("INFO: a LOCTOP file was detected")
                if self.is_chg and self.is_locpot:
                    if self.verbose == "debug":
                        print(
                            "DEBUG: The number of grid points is not what I was expecting. "
                            "The data I got is:"
                        )
                        print(self.Data0[:30])
                        print(self.Data0[-30:])
                    raise RuntimeError(
                        "The file is flagged as a CHGCAR-like file"
                        " and as a LOCPOT at the same time. This is inconsistent"
                    )
            else:
                raise RuntimeError("Grid points do not agree")
        else:
            self.Data0 = np.array(data0_split, dtype=np.float64).reshape(ngfz, ngfy, ngfx)

        if self.is_chg:
            self.Data0 = self.Data0 / Ndata
            if self.verbose:
                print("Total charge", np.sum(self.Data0))

        if self.Ispin > 1:
            data1_split = data[1].split()
            if len(data1_split) != Ndata:
                raise RuntimeError("Grid points do not agree")
            self.Data1 = np.array(data1_split, dtype=np.float64).reshape(ngfz, ngfy, ngfx)
            if self.is_chg:
                self.Data1 = self.Data1 / Ndata
                if self.verbose and self.Ispin == 2:
                    print("INFO: total magnetization,", np.sum(self.Data1))

        if self.Ispin == 4:
            data2_split = data[2].split()
            data3_split = data[3].split()
            if len(data2_split) != Ndata or len(data3_split) != Ndata:
                raise RuntimeError("Grid points do not agree")
            self.Data2 = np.array(data2_split, dtype=np.float64).reshape(ngfz, ngfy, ngfx)
            self.Data3 = np.array(data3_split, dtype=np.float64).reshape(ngfz, ngfy, ngfx)
            if self.is_chg:
                self.Data2 = self.Data2 / Ndata
                self.Data3 = self.Data3 / Ndata


class Chg:
    chg: Chg_base
    filename: str
    is_chg: bool
    verbose: bool | str

    def __init__(
        self, filename: str = "CHG", is_chg: bool = True, verbose: bool | str = False
    ) -> None:
        self.chg = Chg_base()
        self.filename = filename
        self.is_chg = is_chg
        self.verbose = verbose
        self.chg.Load(filename=self.filename, frame=0, is_chg=self.is_chg, verbose=self.verbose)

    def _get_data_for_spin(self, spin: int) -> npt.NDArray[np.float64]:
        """Return the data array for the given spin channel."""
        if spin == 0:
            return self.chg.Data0
        if spin == 1:
            return self.chg.Data1
        if spin == 2:
            return self.chg.Data2
        if spin == 3:
            return self.chg.Data3
        raise RuntimeError("No such spin channel, " + str(spin))

    def Zplot(
        self,
        level: int | None = None,
        spin: int = 0,
        cart_level: float | None = None,
        direct_level: float | None = None,
    ) -> Figure:
        """It plots the CHG-like file at an specific z-value, given by
        level. Only works properly when the Z-axis (c-vector) is perpendicular to the
        other axes.

        args:

        spin: the spin channel (i.e. the first `0`, or the second `1`
        entry of the file)

        level: the value of z to plot. In terms of the grid values (see
        NGF in OUTCAR). Only one among `level`, `cart_level`, and
        `direct_level` should be provided.

        cart_level: as `level`, but the value of z is in cartesian.

        direct_level: as `level`, but the value is in direct coordinates.

        """
        if (level and cart_level) or (level and direct_level) or (direct_level and cart_level):
            raise RuntimeError(
                "only one among `level`, `direct_level`, and `cart_level` has to be provided"
            )

        assert self.chg.poscar is not None
        assert self.chg.poscar.lat is not None

        # Initialize with defaults (midpoint of axis)
        level_floor: int = int(self.chg.ngf[2]) // 2
        level_ceil: int = level_floor
        delta_floor: float = 0
        delta_ceil: float = 1

        # setting the different kind of levels in order, `cart_level` sets
        # `direct_level` and so on
        if cart_level is not None:
            c = float(np.linalg.norm(self.chg.poscar.lat[2]))
            direct_level = cart_level / c
            if self.verbose:
                print("INFO: cart_level,", cart_level)
        if direct_level is not None:
            # going to a grid level, crude interpolation of the level
            level_float = direct_level * int(self.chg.ngf[2])
            level_floor = math.floor(level_float)
            level_ceil = math.ceil(level_float)
            delta_floor = 1 - abs(level_float - level_floor)
            delta_ceil = 1 - abs(level_ceil - level_float)
            if level_floor == level_ceil:
                delta_floor, delta_ceil = 1, 0
            if self.verbose == "debug":
                print("INFO: direct_level", direct_level)
        if level is not None:
            # in this case the crude interpolation shouldn't do anything
            level_floor = level
            level_ceil = level
            delta_floor, delta_ceil = 0, 1
            if self.verbose == "debug":
                print("INFO: level", level)
        if level is None and cart_level is None and direct_level is None:
            if self.verbose == "debug":
                print("INFO: default is the midpoint of the axis", level_floor)

        data = self._get_data_for_spin(spin)

        zcolor = data[level_floor] * delta_floor + data[level_ceil] * delta_ceil

        Agrid, Bgrid = np.mgrid[0 : 1 : int(self.chg.ngf[0]) * 1j, 0 : 1 : int(self.chg.ngf[1]) * 1j]
        # cartesian value of each point of the grid
        xgrid = Agrid * self.chg.poscar.lat[0, 0] + Bgrid * self.chg.poscar.lat[1, 0]
        ygrid = Agrid * self.chg.poscar.lat[0, 1] + Bgrid * self.chg.poscar.lat[1, 1]
        if self.verbose == "debug":
            print("DEBUG: Agrid.shape", Agrid.shape)
            print("DEBUG: lattice\n", self.chg.poscar.lat)
            print("DEBUG: xgrid.shape", xgrid.shape)

        xmin, xmax = xgrid.min(), xgrid.max()
        ymin, ymax = ygrid.min(), ygrid.max()
        xi, yi = np.mgrid[
            xmin : xmax : int(self.chg.ngf[0]) * 4j, ymin : ymax : int(self.chg.ngf[1]) * 4j
        ]
        points = np.vstack((xgrid.flatten(), ygrid.flatten())).T
        values = zcolor.flatten()
        if self.verbose == "debug":
            print("DEBUG: points.shape", points.shape)
            print("DEBUG: values.shape", values.shape)
            print("DEBUG: xi.shape", xi.shape)
            print("DEBUG: yi.shape", yi.shape)
        zi = griddata(points, values, (xi, yi), method="linear")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        p1 = ax.pcolormesh(xi, yi, zi, cmap="seismic")
        fig.colorbar(p1)
        ax.set_aspect("equal")
        return fig

    def CutPlot(
        self,
        level: int | None = None,
        spin: int = 0,
        cart_level: float | None = None,
        direct_level: float | None = None,
        axis: str = "c",
    ) -> Figure:
        """It plots the CHG-like file at an specific value of `axis`, given by
        `level`. Only works properly when the selected `axis` (basis vector) is perpendicular
        to the other axes.

        args:

        spin: the spin channel (i.e. the first `0`, or the second `1`
        entry of the file)

        level: the value of x to plot. In terms of the grid values (see
        NGF in OUTCAR). Only one among `level`, `cart_level`, and
        `direct_level` should be provided.

        cart_level: as `level`, but the value of z is in cartesian.

        direct_level: as `level`, but the value is in direct coordinates.

        axis: 'a', 'b', 'c', the basis vector to fix its value

        """
        assert self.chg.poscar is not None
        assert self.chg.poscar.lat is not None

        # an utilitary dict to choose the desired axes
        ax_dict: dict[str, int] = {"a": 0, "b": 1, "c": 2}
        # axis 3 is the axis to make the cut
        ax3 = ax_dict[axis]
        ax1 = int(np.remainder(ax_dict[axis] + 1, 3))
        ax2 = int(np.remainder(ax_dict[axis] + 2, 3))
        if self.verbose:
            print("INFO: Selected axis to cut:", ax3)
        if self.verbose == "debug":
            print("DEBUG: The other axes are:", ax1, ax2)

        if (level and cart_level) or (level and direct_level) or (direct_level and cart_level):
            raise RuntimeError(
                "only one among `level`, `direct_level`, and `cart_level` has to be provided"
            )

        # Initialize with defaults (midpoint of axis)
        level_floor: int = int(self.chg.ngf[ax3]) // 2
        level_ceil: int = level_floor
        delta_floor: float = 0
        delta_ceil: float = 1

        # setting the different kind of levels in order, `cart_level` sets
        # `direct_level` and so on
        if cart_level is not None:
            # length of the perpendicular vector
            L = float(np.linalg.norm(self.chg.poscar.lat[ax3]))
            direct_level = cart_level / L
            if self.verbose:
                print("INFO: cart_level,", cart_level)
        if direct_level is not None:
            # going to a grid level, crude interpolation of the level
            level_float = direct_level * int(self.chg.ngf[ax3])
            level_floor = math.floor(level_float)
            level_ceil = math.ceil(level_float)
            delta_floor = 1 - abs(level_float - level_floor)
            delta_ceil = 1 - abs(level_ceil - level_float)
            if level_floor == level_ceil:
                delta_floor, delta_ceil = 1, 0
            if self.verbose:
                print("INFO: direct_level", direct_level)
        if level is not None:
            # in this case the crude interpolation shouldn't do anything
            level_floor = level
            level_ceil = level
            delta_floor, delta_ceil = 0, 1
            if self.verbose == "debug":
                print("INFO: level", level)

        if level is None and cart_level is None and direct_level is None:
            if self.verbose == "debug":
                print("INFO: default is the midpoint of the axis", level_floor)

        data = self._get_data_for_spin(spin)

        # building a grid with existent points
        zcolor: npt.NDArray[np.float64]
        Agrid: npt.NDArray[np.float64]
        Bgrid: npt.NDArray[np.float64]
        xgrid: npt.NDArray[np.float64]
        ygrid: npt.NDArray[np.float64]

        if axis == "c":
            zcolor = data[level_floor] * delta_floor + data[level_ceil] * delta_ceil
            # a regular orthogonal grid
            Agrid, Bgrid = np.mgrid[
                0 : 1 : int(self.chg.ngf[0]) * 1j, 0 : 1 : int(self.chg.ngf[1]) * 1j
            ]
            # cartesian grid
            xgrid = Agrid * self.chg.poscar.lat[0, 0] + Bgrid * self.chg.poscar.lat[1, 0]
            ygrid = Agrid * self.chg.poscar.lat[0, 1] + Bgrid * self.chg.poscar.lat[1, 1]

        elif axis == "b":
            zcolor = data[:, level_floor, :] * delta_floor + data[:, level_ceil, :] * delta_ceil
            Agrid, Bgrid = np.mgrid[
                0 : 1 : int(self.chg.ngf[2]) * 1j, 0 : 1 : int(self.chg.ngf[0]) * 1j
            ]
            xgrid = Agrid * self.chg.poscar.lat[2, 2] + Bgrid * self.chg.poscar.lat[0, 2]
            ygrid = Agrid * self.chg.poscar.lat[2, 0] + Bgrid * self.chg.poscar.lat[0, 0]

        elif axis == "a":
            zcolor = data[:, :, level_floor] * delta_floor + data[:, :, level_ceil] * delta_ceil
            Agrid, Bgrid = np.mgrid[
                0 : 1 : int(self.chg.ngf[1]) * 1j, 0 : 1 : int(self.chg.ngf[2]) * 1j
            ]
            xgrid = Agrid * self.chg.poscar.lat[1, 1]
            ygrid = Bgrid * self.chg.poscar.lat[2, 2] + Agrid * self.chg.poscar.lat[1, 0]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        if self.verbose == "debug":
            print("DEBUG: zcolor.shape", zcolor.shape)
            print("DEBUG: Agrid.shape", Agrid.shape)
            print("DEBUG: Bgrid.shape", Bgrid.shape)
            print("DEBUG: lattice\n", self.chg.poscar.lat)
            print("DEBUG: xgrid.shape", xgrid.shape)
            print("DEBUG: ygrid.shape", xgrid.shape)

        xmin, xmax = xgrid.min(), xgrid.max()
        ymin, ymax = ygrid.min(), ygrid.max()
        # creating a regular grid to interpolate (the `4` is the resolution)
        xi, yi = np.mgrid[
            xmin : xmax : int(self.chg.ngf[ax1]) * 4j, ymin : ymax : int(self.chg.ngf[ax2]) * 4j
        ]
        points = np.vstack((xgrid.flatten(), ygrid.flatten())).T
        values = zcolor.flatten()
        if self.verbose == "debug":
            print("points.shape", points.shape)
            print("values.shape", values.shape)
            print("xi.shape", xi.shape)
            print("yi.shape", yi.shape)
        zi = griddata(points, values, (xi, yi), method="linear")

        fig = plt.figure()
        ax_plot = fig.add_subplot(111)
        p1 = ax_plot.pcolormesh(xi, yi, zi, cmap="seismic")
        fig.colorbar(p1)
        ax_plot.set_aspect("equal")
        return fig

    def shift(
        self,
        _x: float | None = None,
        _y: float | None = None,
        _z: float | None = None,
    ) -> None:
        pass

    def plot_atoms(self) -> None:
        _positions = self.chg.poscar.cpos if self.chg.poscar else None

    def plot_cut_new(self, _axis: npt.NDArray[np.float64], _value: float | None) -> None:
        """Temporary function to use the plot3D class... very limited
        functionality for now

        """
        assert self.chg.poscar is not None
        assert self.chg.poscar.lat is not None
        p3d = plot3d.data3D(data=self.chg.Data0, lattice=self.chg.poscar.lat, verbose="debug")
        p3d.cut_plane(axis=np.array([0.0, 0.0, 1.0]), value=10)

    def average(self, axis: str) -> None:
        assert self.chg.poscar is not None
        assert self.chg.poscar.lat is not None

        data: npt.NDArray[np.float64] = self.chg.Data0
        length: float
        if axis == "c":
            data = np.asarray(np.average(data, axis=2), dtype=np.float64)
            data = np.asarray(np.average(data, axis=1), dtype=np.float64)
            length = float(np.linalg.norm(self.chg.poscar.lat[2]))
        elif axis == "b":
            data = np.asarray(np.average(data, axis=2), dtype=np.float64)
            data = np.asarray(np.average(data, axis=0), dtype=np.float64)
            length = float(np.linalg.norm(self.chg.poscar.lat[1]))
        elif axis == "a":
            data = np.asarray(np.average(data, axis=1), dtype=np.float64)
            data = np.asarray(np.average(data, axis=0), dtype=np.float64)
            length = float(np.linalg.norm(self.chg.poscar.lat[0]))
        else:
            raise ValueError(f"Invalid axis: {axis}")
        print("Averaged Data shape, ", data.shape)
        x = np.linspace(0, length, len(data))
        plt.plot(x, data)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, help="input file (CHGCAR, CHG, ELFCAR or LOCPOT)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true", help="Even more verbosity")
    parser.add_argument(
        "-n",
        "--no_scale",
        action="store_true",
        help="Set when openning a ELFCAR or LOCPOT. The CHGCAR"
        " uses a different normalization, this flag avoids it.",
    )
    parser.add_argument("-a", "--axis", choices=["a", "b", "c"], default="c", help="Axis to cut")
    parser.add_argument("-z", action="store_true", help="Fallback utility function to plot")

    parser.add_argument("--new", action="store_true", help="usage of new, not fully tested methods")
    parser.add_argument(
        "-p",
        "--average",
        action="store_true",
        help="averages the potential (or charge) along a given axis",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--direct_level", type=float, help="level to plot, in direct coords")
    group.add_argument(
        "-c", "--cartesian_level", type=float, help="level to plot, in cartesian coords"
    )

    args = parser.parse_args()

    verbose_val: bool | str = bool(args.verbose)
    if args.debug:
        verbose_val = "debug"

    inputfile: str = args.inputfile
    no_scale: bool = args.no_scale
    axis_val: str = args.axis
    z_flag: bool = args.z
    new_flag: bool = args.new
    average_flag: bool = args.average
    direct_level_val: float | None = args.direct_level
    cartesian_level_val: float | None = args.cartesian_level

    if "POT" in inputfile or "ELF" in inputfile:
        if no_scale is False:
            warnings.warn(
                "It seems you are openning a LOCPOT or ELFCAR file."
                " If so, you should add the option '-n' to get the "
                "correct scaling"
            )
    elif "CHG" in inputfile and no_scale is True:
        warnings.warn(
            "It seems you are openning a CHG or CHGCAR file."
            " If so, you should not add the option '-n' to get the "
            "correct scaling"
        )
    is_chg = not no_scale

    chg = Chg(filename=inputfile, is_chg=is_chg, verbose=verbose_val)

    fig: Figure
    if z_flag:
        fig = chg.Zplot(cart_level=cartesian_level_val, direct_level=direct_level_val)
    elif new_flag:
        chg.plot_cut_new(_value=direct_level_val, _axis=np.array([0.0, 0.0, 1.0]))
    elif average_flag:
        chg.average(axis=axis_val)
    else:
        fig = chg.CutPlot(
            cart_level=cartesian_level_val, direct_level=direct_level_val, axis=axis_val
        )
    plt.show()
