"""
Created on Fri May 10 16:23:30 2019

@author: Pedram Tavadze

"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.spatial import ConvexHull, Voronoi
from skimage import measure

from ..core import ProcarSelect
from ..io import ProcarParser
from ..utils.utilsprocar import UtilsProcar
from .splash import welcome


def get_wigner_seitz(recLat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    kpoints: list[npt.NDArray[np.float64]] = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                vec = i * recLat[0] + j * recLat[1] + k * recLat[2]
                kpoints.append(vec)
    brill = Voronoi(np.array(kpoints))
    faces: list[list[int]] = []
    for idict in brill.ridge_dict:
        if idict[0] == 13 or idict[1] == 13:
            faces.append(brill.ridge_dict[idict])
    verts = brill.vertices
    poly: list[npt.NDArray[np.float64]] = []
    for ix in range(len(faces)):
        temp: list[npt.NDArray[np.float64]] = []
        for iy in range(len(faces[ix])):
            temp.append(verts[faces[ix][iy]])
        poly.append(np.array(temp))
    return np.array(poly)


def mapping_func(
    kpoints: npt.NDArray[np.float64], bands: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    kx = np.unique(kpoints[:, 0])
    ky = np.unique(kpoints[:, 1])
    kz = np.unique(kpoints[:, 2])
    mapped_func: npt.NDArray[np.float64] = np.zeros(shape=(len(kx), len(ky), len(kz)))
    kpoint_matrix: npt.NDArray[np.float64] = np.zeros(shape=(len(kx), len(ky), len(kz), 3))
    for ikx in range(len(kx)):
        cond1 = kpoints[:, 0] == kx[ikx]
        for iky in range(len(ky)):
            cond2 = kpoints[:, 1] == ky[iky]
            for ikz in range(len(kz)):
                cond3 = kpoints[:, 2] == kz[ikz]
                tot_cond = np.all([cond1, cond2, cond3], axis=0)
                if len(bands[tot_cond]) != 0:
                    mapped_func[ikx, iky, ikz] = bands[tot_cond][0]
                    kpoint_matrix[ikx, iky, ikz] = [kx[ikx], ky[iky], kz[ikz]]
                else:
                    mapped_func[ikx, iky, ikz] = np.nan
                    kpoint_matrix[ikx, iky, ikz] = [np.nan, np.nan, np.nan]

    return mapped_func, kpoint_matrix


# if the grid is gamma centered the gird would be something like [-0.45,-0.35,...,0,...,0.5]
# using FFT interpolate we will loose center. The fermi surface will not be symmetric with
# respect to the center. To avoid this we will add the points from 0.5 to -0.5 so the mesh will be like
# [-0.5,-0.45,..,0,...,0.45,0.5]
def symmetrize(data: ProcarSelect) -> ProcarSelect:
    assert data.kpoints is not None
    assert data.bands is not None
    assert data.spd is not None
    # kpoints with one 0.5
    idx = (data.kpoints == 0.5).sum(axis=1) == 1
    #    idx = np.any(data.kpoints == 0.5,axis=1)
    kpoints_toAdd = data.kpoints[idx] + (data.kpoints[idx] == 0.5).astype(int) * -1
    bands_toAdd = data.bands[idx]
    spd_toAdd = data.spd[idx]
    # kpoints with two 0.5
    # -0.5,-0.5
    idx = (data.kpoints == 0.5).sum(axis=1) == 2
    kpoints_temp = data.kpoints[idx] + (data.kpoints[idx] == 0.5).astype(int) * -1
    bands_temp = data.bands[idx]
    spd_temp = data.spd[idx]

    kpoints_toAdd = np.append(kpoints_toAdd, kpoints_temp, axis=0)
    bands_toAdd = np.append(bands_toAdd, bands_temp, axis=0)
    spd_toAdd = np.append(spd_toAdd, spd_temp, axis=0)
    # [0.5,:,:]
    kpoints_temp = data.kpoints[idx][((data.kpoints[idx] == 0.5)[:, 0])] - [1, 0, 0]
    bands_temp = data.bands[idx][((data.kpoints[idx] == 0.5)[:, 0])]
    spd_temp = data.spd[idx][((data.kpoints[idx] == 0.5)[:, 0])]
    kpoints_toAdd = np.append(kpoints_toAdd, kpoints_temp, axis=0)
    bands_toAdd = np.append(bands_toAdd, bands_temp, axis=0)
    spd_toAdd = np.append(spd_toAdd, spd_temp, axis=0)
    # [:,0.5,:]
    kpoints_temp = data.kpoints[idx][((data.kpoints[idx] == 0.5)[:, 1])] - [0, 1, 0]
    bands_temp = data.bands[idx][((data.kpoints[idx] == 0.5)[:, 1])]
    spd_temp = data.spd[idx][((data.kpoints[idx] == 0.5)[:, 1])]
    kpoints_toAdd = np.append(kpoints_toAdd, kpoints_temp, axis=0)
    bands_toAdd = np.append(bands_toAdd, bands_temp, axis=0)
    spd_toAdd = np.append(spd_toAdd, spd_temp, axis=0)
    # [:,:,0.5]
    kpoints_temp = data.kpoints[idx][((data.kpoints[idx] == 0.5)[:, 2])] - [0, 0, 1]
    bands_temp = data.bands[idx][((data.kpoints[idx] == 0.5)[:, 2])]
    spd_temp = data.spd[idx][((data.kpoints[idx] == 0.5)[:, 2])]
    kpoints_toAdd = np.append(kpoints_toAdd, kpoints_temp, axis=0)
    bands_toAdd = np.append(bands_toAdd, bands_temp, axis=0)
    spd_toAdd = np.append(spd_toAdd, spd_temp, axis=0)
    # kpoints with three 0.5
    idx = (data.kpoints == 0.5).sum(axis=1) == 3
    kpoints_temp = np.array(
        [(x, y, z) for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]
    )[:-1]
    bands_temp = np.repeat(data.bands[idx], 7, axis=0)
    spd_temp = np.repeat(data.spd[idx], 7, axis=0)
    kpoints_toAdd = np.append(kpoints_toAdd, kpoints_temp, axis=0)
    bands_toAdd = np.append(bands_toAdd, bands_temp, axis=0)
    spd_toAdd = np.append(spd_toAdd, spd_temp, axis=0)

    data.kpoints = np.append(data.kpoints, kpoints_toAdd, axis=0)
    data.bands = np.append(data.bands, bands_toAdd, axis=0)
    data.spd = np.append(data.spd, spd_toAdd, axis=0)
    return data


def bring_pnts_to_BZ(
    recLat: npt.NDArray[np.float64],
    kvector_cart: npt.NDArray[np.float64],
    kvector_red: npt.NDArray[np.float64],
    br_points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    outsides: list[npt.NDArray[np.float64]] = []
    directions: list[npt.NDArray[np.float64]] = []
    # This section finds points that are outside of the 1st BZ and and creates those points in the 1st BZ
    movements = np.array(
        [
            recLat[0],
            recLat[1],
            recLat[2],
            -1 * recLat[0],
            -1 * recLat[1],
            -1 * recLat[2],
        ]
    )
    for ik in range(len(kvector_cart)):
        ik_copy = kvector_cart[ik][:]
        idirection = 0
        was_outside = False
        while is_outside([br_points, ik_copy]) and idirection < 6:
            ik_copy = kvector_cart[ik] + movements[idirection]
            idirection += 1
            was_outside = True
        if was_outside:
            outsides.append(kvector_red[ik])
            directions.append(np.dot(movements[idirection - 1], np.linalg.pinv(recLat)).round(2))
        kvector_cart[ik] = ik_copy[:]
        kvector_red[ik] = np.dot(ik_copy, np.linalg.pinv(recLat)).round(2)
    if outsides:
        has_points_out = True
    else:
        has_points_out = False
    return kvector_cart, kvector_red, has_points_out


def fft_interpolate(
    function: npt.NDArray[np.float64], scale: int
) -> npt.NDArray[np.float64]:
    eigen_fft = np.fft.fftn(function)
    shifted_fft = np.fft.fftshift(eigen_fft)
    nx, ny, nz = np.array(shifted_fft.shape)
    pad_x = nx * (scale - 1) // 2
    pad_y = ny * (scale - 1) // 2
    pad_z = nz * (scale - 1) // 2
    new_matrix = np.pad(
        shifted_fft,
        ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)),
        "constant",
        constant_values=0,
    )

    new_matrix = np.fft.ifftshift(new_matrix)
    interpolated = np.real(np.fft.ifftn(new_matrix)) * (scale**3)
    return interpolated


def is_outside(args: list[Any]) -> bool:
    br_points = args[0]
    point = args[1]
    Hull1 = ConvexHull(br_points)
    added_points = np.append(br_points, [point], axis=0)
    Hull2 = ConvexHull(added_points)
    if len(Hull2.vertices) == len(Hull1.vertices):
        if sum(Hull2.vertices - Hull1.vertices) != 0:
            return True
    else:
        return True
    return False


def to_remove(args: list[Any]) -> npt.NDArray[np.bool_]:
    faces = args[0]
    vert = args[1]
    return np.any(faces == vert, axis=1)


def fermi3D(
    procar: str,
    outcar: str,
    bands: int | range = -1,
    scale: int = 1,
    mode: str = "plain",
    st: bool = False,
    **kwargs: Any,
) -> None:
    """
    This function plots 3d fermi surface
    list of acceptable kwargs :
        plotting_package
        nprocess
        face_colors
        arrow_colors
        arrow_spin
        atom
        orbital
        spin

    """
    welcome()

    # Initilizing the arguments :

    if "plotting_package" in kwargs:
        plotting_package: str = kwargs["plotting_package"]
    else:
        plotting_package = "mayavi"

    if "nprocess" in kwargs:
        nprocess: int = kwargs["nprocess"]
    else:
        nprocess = 2

    if "face_colors" in kwargs:
        face_colors: Any = kwargs["face_colors"]
    else:
        face_colors = None
    if "cmap" in kwargs:
        cmap: Any = kwargs["cmap"]
    else:
        cmap = "jet"
    if "atoms" in kwargs:
        atoms: list[int] = kwargs["atoms"]
    else:
        atoms = [-1]  # project all atoms
    if "orbitals" in kwargs:
        orbitals: list[int] = kwargs["orbitals"]
    else:
        orbitals = [-1]

    if "spin" in kwargs:
        spin: list[int] = kwargs["spin"]
    else:
        spin = [0]
    if "mask_points" in kwargs:
        mask_points: int = kwargs["mask_points"]
    else:
        mask_points = 1
    if "energy" in kwargs:
        energy: float = kwargs["energy"]
    else:
        energy = 0
    if "transparent" in kwargs:
        transparent: bool = kwargs["transparent"]
    else:
        transparent = False

    if "arrow_projection" in kwargs:
        arrow_projection: int = kwargs["arrow_projection"]
    else:
        arrow_projection = 2

    # Initialize plotting library references (set conditionally below)
    mlab: Any = None
    tvtk: Any = None
    ff: Any = None
    go: Any = None
    plt: Any = None
    Poly3DCollection: Any = None
    ipv: Any = None
    figs: list[Any] = []
    ax: Any = None

    if plotting_package == "mayavi":
        try:
            from mayavi import mlab as _mlab  # pyright: ignore[reportMissingModuleSource]
            from tvtk.api import tvtk as _tvtk  # pyright: ignore[reportMissingModuleSource]

            mlab = _mlab
            tvtk = _tvtk
        except Exception:
            print(
                "You have selected mayavi as plottin package. please install mayavi and tvtk or choose a different package"
            )
            return
    elif plotting_package == "plotly":
        try:
            import matplotlib.cm as mcm
            import plotly.figure_factory as _ff  # pyright: ignore[reportMissingModuleSource]
            import plotly.graph_objs as _go  # pyright: ignore[reportMissingModuleSource]

            ff = _ff
            go = _go
            cmap = mcm.get_cmap(cmap)
            figs = []

        except Exception:
            print(
                "You have selected plotly as plottin package. please install plotly or choose a different package"
            )
            return
    elif plotting_package == "matplotlib":
        try:
            import matplotlib.pylab as _plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection

            plt = _plt
            Poly3DCollection = _Poly3DCollection
        except Exception:
            print(
                "You have selected matplotlib as plotting package. please install matplotlib or choose a different package"
            )
            return
    elif plotting_package == "ipyvolume":
        try:
            import ipyvolume.pylab as _ipv  # pyright: ignore[reportMissingModuleSource]

            ipv = _ipv
        except Exception:
            print(
                "You have selected ipyvolume as plotting package. please install ipyvolume or choose a different package"
            )
            return
    if mode == "colorful":
        face_colors = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (0, 1, 1),
            (1, 0, 1),
            (192 / 255, 192 / 255, 192 / 255),
            (128 / 255, 128 / 255, 128 / 255),
            (128 / 255, 0, 0),
            (128 / 255, 128 / 255, 0),
            (0, 128 / 255, 0),
            (128 / 255, 0, 128 / 255),
            (0, 128 / 255, 128 / 255),
            (0, 0, 128 / 255),
        ]

    permissive = False

    # get fermi from outcar
    outcarparser = UtilsProcar()
    e_fermi = outcarparser.FermiOutcar(outcar)
    print("Fermi=", e_fermi)
    e_fermi += energy
    # get reciprocal lattice from outcar
    recLat = outcarparser.RecLatOutcar(outcar)

    # parsing the Procar file
    procarFile = ProcarParser()
    procarFile.readFile(procar, permissive)

    poly = get_wigner_seitz(recLat)
    # plot brilliouin zone
    if plotting_package == "mayavi":
        brillouin_point: list[list[float]] = []
        brillouin_faces: list[list[int]] = []
        point_count = 0
        for iface in poly:
            single_face: list[int] = []
            for ipoint in iface:
                single_face.append(point_count)
                brillouin_point.append(list(ipoint))
                point_count += 1
            brillouin_faces.append(single_face)
        polydata_br = tvtk.PolyData(points=brillouin_point, polys=brillouin_faces)
        mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(400, 350))
        mlab.pipeline.surface(
            polydata_br,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=4,
            name="BRZ",
        )
    elif plotting_package == "plotly":
        for iface_arr in poly:
            iface_arr = np.pad(iface_arr, ((0, 1), (0, 0)), "wrap")
            x, y, z = iface_arr[:, 0], iface_arr[:, 1], iface_arr[:, 2]
            plane = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="black", width=4))
            figs.append(plane)

    elif plotting_package == "matplotlib":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        brillouin_zone = Poly3DCollection(
            poly, facecolors=["None"] * len(poly), alpha=1, linewidth=4
        )
        brillouin_zone.set_edgecolor("k")
        ax.add_collection3d(brillouin_zone, zs=0, zdir="z")

    br_points: list[Any] = []
    for iface_arr in poly:
        for ipoint in iface_arr:
            br_points.append(ipoint)
    br_points_arr: npt.NDArray[np.float64] = np.unique(br_points, axis=0)
    print("Number of bands: %d" % procarFile.bandsCount)
    print("Number of koints %d" % procarFile.kpointsCount)
    print("Number of ions: %d" % procarFile.ionsCount)
    print("Number of orbitals: %d" % procarFile.orbitalCount)
    print("Number of spins: %d" % procarFile.ispin)

    # selecting the data
    # ProcarParser has nullable field types that don't match ProcarDataProtocol
    # at the type level, but after readFile() the fields are populated
    procar_data: Any = procarFile
    data = ProcarSelect(procar_data, deepCopy=True)
    assert data.bands is not None
    assert data.kpoints is not None
    assert data.spd is not None

    band_indices: range
    if isinstance(bands, int):
        band_indices = range(data.bands.shape[1])
    else:
        band_indices = bands

    kvector = data.kpoints
    kmax = np.max(kvector)
    kmin = np.min(kvector)

    if abs(kmax) != abs(kmin):
        print("The mesh provided is gamma center, symmetrizing data")
        print("For a better fermi surface, use a non-gamma centered k-mesh")
        data = symmetrize(data)
        assert data.kpoints is not None
        kvector = data.kpoints

    assert data.kpoints is not None
    kvector_red = data.kpoints.copy()
    kvector_cart = np.dot(kvector_red, recLat)

    # This section finds points that are outside of the 1st BZ and and creates those points in the 1st BZ
    kvector_cart, kvector_red, has_points_out = bring_pnts_to_BZ(
        recLat, kvector_cart, kvector_red, br_points_arr
    )
    #    has_points_out = False

    # getting the mesh grid in each dirrection
    kx_red = np.unique(kvector_red[:, 0])
    ky_red = np.unique(kvector_red[:, 1])
    kz_red = np.unique(kvector_red[:, 2])

    # getting the lengths between kpoints in each direction
    klength_x = np.abs(kx_red[-1] - kx_red[-2])
    klength_y = np.abs(ky_red[-1] - ky_red[-2])
    klength_z = np.abs(kz_red[-1] - kz_red[-2])
    klengths = [klength_x, klength_y, klength_z]

    # getting number of kpoints in each direction with the addition of kpoints needed to sample the 1st BZ fully (in reduced)
    nkx_red = kx_red.shape[0]
    nky_red = ky_red.shape[0]
    nkz_red = kz_red.shape[0]

    # getting numner of kpoints in each direction provided by vasp
    nkx_orig = np.unique(kvector[:, 0]).shape[0]
    nky_orig = np.unique(kvector[:, 1]).shape[0]
    nkz_orig = np.unique(kvector[:, 2]).shape[0]

    # Amount of kpoints needed to add on to fully sample 1st BZ
    padding_x = (nkx_red - nkx_orig) // 2
    padding_y = (nky_red - nky_orig) // 2
    padding_z = (nkz_red - nkz_orig) // 2

    # Initialize mode-specific variables (set conditionally below)
    color_kvector_cart: npt.NDArray[np.float64] | None = None
    color_eigen: list[list[float]] = []
    colors: npt.NDArray[np.floating[Any]] | None = None
    spin_arrows: npt.NDArray[np.floating[Any]] | None = None
    centers: npt.NDArray[np.float64] | None = None
    dataX: ProcarSelect | None = None
    dataY: ProcarSelect | None = None
    dataZ: ProcarSelect | None = None

    if mode == "parametric":
        data.selectIspin(spin)
        data.selectAtoms(atoms, fortran=False)
        data.selectOrbital(orbitals)
    elif mode == "external":
        if "color_file" in kwargs:
            rf = open(kwargs["color_file"])

            lines = rf.readlines()
            counter = 0
            color_kvector: list[list[float]] = []
            for iline in lines:
                if counter < 2:
                    if "band" in iline:
                        counter += 1
                        continue
                    temp_floats = [float(x) for x in iline.split()]
                    color_kvector.append([temp_floats[0], temp_floats[1], temp_floats[2]])
            counter = -1
            for iline in lines:
                if "band" in iline:
                    counter += 1
                    _iband = int(iline.split()[-1])
                    color_eigen.append([])
                    continue
                color_eigen[counter].append(float(iline.split()[-1]))
            rf.close()

            color_kvector_np = np.array(color_kvector)
            color_kvector_red = color_kvector_np.copy()
            color_kvector_cart = np.dot(color_kvector_np, recLat)
            if has_points_out:
                assert color_kvector_cart is not None
                color_kvector_cart, color_kvector_red, _temp_flag = bring_pnts_to_BZ(
                    recLat, color_kvector_cart, color_kvector_red, br_points_arr
                )
        else:
            print("mode selected was external, but no color_file name was provided")
            return
    if st:
        dataX = ProcarSelect(procar_data, deepCopy=True)
        dataY = ProcarSelect(procar_data, deepCopy=True)
        dataZ = ProcarSelect(procar_data, deepCopy=True)

        dataX.kpoints = data.kpoints
        dataY.kpoints = data.kpoints
        dataZ.kpoints = data.kpoints

        dataX.spd = data.spd
        dataY.spd = data.spd
        dataZ.spd = data.spd

        dataX.bands = data.bands
        dataY.bands = data.bands
        dataZ.bands = data.bands

        dataX.selectIspin([1])
        dataY.selectIspin([2])
        dataZ.selectIspin([3])

        dataX.selectAtoms(atoms, fortran=False)
        dataY.selectAtoms(atoms, fortran=False)
        dataZ.selectAtoms(atoms, fortran=False)

        dataX.selectOrbital(orbitals)
        dataY.selectOrbital(orbitals)
        dataZ.selectOrbital(orbitals)
    ic = 0
    for iband in band_indices:
        print("Plotting band %d" % iband)

        assert data.bands is not None
        assert data.spd is not None
        assert data.kpoints is not None
        eigen = data.bands[:, iband]

        # mapping the eigen values on the mesh grid to a matrix
        mapped_func, _kpoint_matrix = mapping_func(kvector, eigen)

        # adding the points from the 2nd BZ to 1st BZ to fully sample the BZ. Check np.pad("wrap") for more information
        mapped_func = np.pad(
            mapped_func,
            ((padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z)),
            "wrap",
        )

        # Fourier interpolate the mapped function E(x,y,z)
        surf_equation = fft_interpolate(mapped_func, scale)

        # after the FFT we loose the center of the BZ, using numpy roll we bring back the center of the BZ
        surf_equation = np.roll(surf_equation, (scale) // 2, axis=[0, 1, 2])

        try:
            # creating the isosurface if possible
            verts, faces, _normals, _values = measure.marching_cubes_lewiner(surf_equation, e_fermi)

        except Exception:
            print("No isosurface for this band")
            continue
        # the vertices provided are scaled and shifted to start from zero
        # we center them to zero, and rescale them to fit the real BZ by multiplying by the klength in each direction
        for ix in range(3):
            verts[:, ix] -= verts[:, ix].min()
            verts[:, ix] -= (verts[:, ix].max() - verts[:, ix].min()) / 2
            verts[:, ix] *= klengths[ix] / scale

        # the vertices need to be transformed to reciprocal spcae from recuded space, to find the points that are
        # in 2nd BZ, to be removed
        verts = np.dot(verts, recLat)

        # identifying the points in 2nd BZ and removing them
        if has_points_out:
            args: list[list[Any]] = []
            for ivert in range(len(verts)):
                args.append([br_points_arr, verts[ivert]])

            p = Pool(nprocess)
            results = np.array(p.map(is_outside, args))
            p.close()
            out_verts = np.arange(0, len(results))[results]
            new_faces: list[npt.NDArray[np.intp]] = []
            #            outs_bool_mat = np.zeros(shape=faces.shape,dtype=np.bool)

            for iface_row in faces:
                remove = False
                for ivert in iface_row:
                    if ivert in out_verts:
                        remove = True

                        continue

                if not remove:
                    new_faces.append(iface_row)
            faces = np.array(new_faces)

        print("done removing")
        # At this point we have the plain Fermi surface, we can color the surface depending on the projection
        # We create the center of faces by averaging coordinates of corners

        if mode == "parametric":
            character = data.spd[:, iband]

            centers = np.zeros(shape=(len(faces), 3))
            for iface_idx in range(len(faces)):
                centers[iface_idx, 0:3] = np.average(verts[faces[iface_idx]], axis=0)

            colors = interpolate.griddata(kvector_cart, character, centers, method="nearest")
        elif mode == "external":
            character = np.array(color_eigen[ic])
            ic += 1
            centers = np.zeros(shape=(len(faces), 3))
            for iface_idx in range(len(faces)):
                centers[iface_idx, 0:3] = np.average(verts[faces[iface_idx]], axis=0)

            assert color_kvector_cart is not None
            colors = interpolate.griddata(color_kvector_cart, character, centers, method="nearest")

        if st:
            assert dataX is not None
            assert dataY is not None
            assert dataZ is not None
            assert dataX.spd is not None
            assert dataY.spd is not None
            assert dataZ.spd is not None
            projection_x = dataX.spd[:, iband]
            projection_y = dataY.spd[:, iband]
            projection_z = dataZ.spd[:, iband]

            verts_spin, faces_spin, _normals2, _values2 = measure.marching_cubes_lewiner(
                mapped_func, e_fermi
            )

            for ix in range(3):
                verts_spin[:, ix] -= verts_spin[:, ix].min()
                verts_spin[:, ix] -= (verts_spin[:, ix].max() - verts_spin[:, ix].min()) / 2
                verts_spin[:, ix] *= klengths[ix]
            verts_spin = np.dot(verts_spin, recLat)

            if has_points_out:
                args = []
                for ivert in range(len(verts_spin)):
                    args.append([br_points_arr, verts_spin[ivert]])

                p = Pool(nprocess)
                results = np.array(p.map(is_outside, args))
                p.close()
                out_verts = np.arange(0, len(results))[results]

                new_faces_spin: list[npt.NDArray[np.intp]] = []
                for iface_row in faces_spin:
                    remove = False
                    for ivert in iface_row:
                        if ivert in out_verts:
                            remove = True
                            continue
                    if not remove:
                        new_faces_spin.append(iface_row)
                faces_spin = np.array(new_faces_spin)

            centers = np.zeros(shape=(len(faces_spin), 3))
            for iface_idx in range(len(faces_spin)):
                centers[iface_idx, 0:3] = np.average(verts_spin[faces_spin[iface_idx]], axis=0)

            colors1 = interpolate.griddata(kvector_cart, projection_x, centers, method="linear")
            colors2 = interpolate.griddata(kvector_cart, projection_y, centers, method="linear")
            colors3 = interpolate.griddata(kvector_cart, projection_z, centers, method="linear")
            spin_arrows = np.vstack((colors1, colors2, colors3)).T

        if plotting_package == "mayavi":
            polydata = tvtk.PolyData(points=verts, polys=faces)

            if face_colors != None:
                mlab.pipeline.surface(
                    polydata,
                    representation="surface",
                    color=face_colors[ic],
                    opacity=1,
                    name="band-" + str(iband),
                )
                ic += 1
            elif mode == "plain":
                if not (transparent):
                    _s = mlab.pipeline.surface(
                        polydata,
                        representation="surface",
                        color=(0, 0.5, 1),
                        opacity=1,
                        name="band-" + str(iband),
                    )

            elif mode == "parametric" or mode == "external":
                assert colors is not None
                cell_data: Any = polydata.cell_data
                cell_data.scalars = colors
                # tvtk scalars have a .name attribute not present on NDArray
                scalars_ref: Any = cell_data.scalars
                scalars_ref.name = "celldata"
                mlab.pipeline.surface(polydata, vmin=0, vmax=colors.max(), colormap=cmap)
                _cb = mlab.colorbar(orientation="vertical")

            if st:
                assert spin_arrows is not None
                assert centers is not None
                x, y, z = list(zip(*centers))
                u, v, w = list(zip(*spin_arrows))

                pnts = mlab.quiver3d(
                    x,
                    y,
                    z,
                    u,
                    v,
                    w,
                    line_width=5,
                    mode="arrow",
                    resolution=25,
                    reset_zoom=False,
                    name="spin-" + str(iband),
                    mask_points=mask_points,
                    scalars=spin_arrows[:, arrow_projection],
                    vmin=-1,
                    vmax=1,
                    colormap=cmap,
                )
                pnts.glyph.color_mode = "color_by_scalar"
                pnts.glyph.glyph_source.glyph_source.shaft_radius = 0.05
                pnts.glyph.glyph_source.glyph_source.tip_radius = 0.1

        elif plotting_package == "plotly":
            if mode == "plain":
                if not (transparent):
                    x, y, z = zip(*verts)
                    plotly_fig = ff.create_trisurf(
                        x=x,
                        y=y,
                        z=z,
                        plot_edges=False,
                        simplices=faces,
                        title="band-%d" % ic,
                    )

                    figs.append(plotly_fig["data"][0])

            elif mode == "parametric" or mode == "external":
                face_colors = cmap(colors)
                colormap_list = [
                    "rgb(%i,%i,%i)" % (x[0], x[1], x[2]) for x in (face_colors * 255).round()
                ]
                x, y, z = zip(*verts)
                plotly_fig = ff.create_trisurf(
                    x=x,
                    y=y,
                    z=z,
                    plot_edges=False,
                    colormap=colormap_list,
                    simplices=faces,
                    show_colorbar=True,
                    title="band-%d" % ic,
                )

                figs.append(plotly_fig["data"][0])
        elif plotting_package == "matplotlib":
            if mode == "plain":
                x, y, z = zip(*verts)
                ax.plot_trisurf(x, y, faces, z, linewidth=0.2, antialiased=True)
            elif mode == "parametric" or mode == "external":
                print(
                    "coloring the faces is not implemented in matplot lib, please use another plotting package.we recomend mayavi."
                )
        elif plotting_package == "ipyvolume":
            if mode == "plain":
                ipv.figure()
                ipv.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces)

            elif mode == "paramteric" or mode == "external":
                face_colors = cmap(colors)
                colormap_list = [
                    "rgb(%i,%i,%i)" % (x[0], x[1], x[2]) for x in (face_colors * 255).round()
                ]
                ipv.figure()
                ipv.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color=cmap)

    # if plotting_package == "mayavi":
    #     mlab.colorbar(orientation="vertical")
    #     mlab.show()
    # elif plotting_package == "plotly":
    #     layout = go.Layout(showlegend=False)

    #     fig = go.Figure(data=figs, layout=layout)
    #     py.iplot(fig)
    # elif plotting_package == "matplotlib":
    #     plt.show()
    # elif plotting_package == "ipyvolume":
    #     ipv.show()

    return
