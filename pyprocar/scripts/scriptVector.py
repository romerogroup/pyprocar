from __future__ import annotations

from pyprocar.core import FermiSurface, ProcarSelect
from pyprocar.io import ProcarParser
from pyprocar.utils.splash import welcome
from pyprocar.utils.utilsprocar import UtilsProcar


def Vector(
    infile: str,
    bands: list[int] | None = None,
    energy: float | None = None,
    fermi: float | None = None,
    atoms: list[int] | None = None,
    orbitals: list[int] | None = None,
    outcar: str | None = None,
    scale: float = 0.1,
    code: str = "vasp",
    repair: bool = True,
) -> None:
    welcome()

    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(infile, infile)
            print("PROCAR repaired. Run with repair=False next time.")

    print("Input File    : ", infile)
    print("Bands         : ", bands)
    print("Energy        : ", energy)
    print("Fermi         : ", fermi)
    print("outcar        : ", outcar)
    print("atoms         : ", atoms)
    print("orbitals      : ", orbitals)
    print("scale factor  : ", scale)

    if bands == [] and energy is None:
        raise RuntimeError("You must provide the bands or energy.")
    if fermi is None and outcar is None:
        print("WARNING: Fermi's Energy not set")

    # first parse the outcar if given
    recLat = None  # Will contain reciprocal vectors, if necessary
    if outcar:
        outcarparser = UtilsProcar()
        if fermi is None:
            fermi = outcarparser.FermiOutcar(outcar)
            # if quiet is False:
            print("Fermi energy found in outcar file = " + str(fermi))
        recLat = outcarparser.RecLatOutcar(outcar)

    if atoms is None:
        atoms = [-1]
    if orbitals is None:
        orbitals = [-1]

    # parsing the file
    procarFile = ProcarParser()
    procarFile.readFile(infile, recLattice=recLat)

    # processing the data
    sx = ProcarSelect(procarFile, deepCopy=True)  # pyright: ignore[reportArgumentType]
    sx.selectIspin([1])
    sx.selectAtoms(atoms)
    sx.selectOrbital(orbitals)

    sy = ProcarSelect(procarFile, deepCopy=True)  # pyright: ignore[reportArgumentType]
    sy.selectIspin([2])
    sy.selectAtoms(atoms)
    sy.selectOrbital(orbitals)

    sz = ProcarSelect(procarFile, deepCopy=True)  # pyright: ignore[reportArgumentType]
    sz.selectIspin([3])
    sz.selectAtoms(atoms)
    sz.selectOrbital(orbitals)

    assert sx.kpoints is not None
    x = sx.kpoints[:, 0]
    y = sx.kpoints[:, 1]
    z = sx.kpoints[:, 2]

    # if energy was given I need to find the bands indexes crossing it
    if energy is not None:
        assert sx.bands is not None
        assert fermi is not None
        FerSurf = FermiSurface(sx.kpoints, sx.bands - fermi, sx.spd)  # pyright: ignore[reportCallIssue]
        FerSurf.FindEnergy(energy)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        bands = list(FerSurf.useful[0])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        print("Bands indexes crossing Energy  ", energy, ", are: ", bands)

    from mayavi import mlab  # pyright: ignore[reportMissingModuleSource]

    _fig = mlab.figure(bgcolor=(1, 1, 1))

    assert bands is not None
    for band in bands:
        assert sx.spd is not None
        assert sy.spd is not None
        assert sz.spd is not None
        # z = sx.bands[:,band]-fermi
        u = sx.spd[:, band]
        v = sy.spd[:, band]
        w = sz.spd[:, band]
        scalar = w

        vect = mlab.quiver3d(
            x,
            y,
            z,
            u,
            v,
            w,
            scale_factor=scale,
            scale_mode="vector",
            scalars=scalar,
            mode="arrow",
            colormap="jet",
        )
        vect.glyph.color_mode = "color_by_scalar"
        vect.scene.parallel_projection = True
        vect.scene.z_plus_view()

        # tube= mlab.plot3d(x,y,z, tube_radius=0.0050, color=(0.5,0.5,0.5))
    mlab.show()
