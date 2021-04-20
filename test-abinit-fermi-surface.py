#!/usr/bin/env python

import pyprocar
from pyprocar.abinitparser import AbinitParser
from pyprocar.procarparser import ProcarParser
from pyprocar.procarselect import ProcarSelect
from pyprocar.utilsprocar import UtilsProcar


# bands

# MgB2
# procar_file = "/Users/uthpala/project-data/PyProcar/fermi-surface/MgB2-bands/PROCAR"
# abinit_file = "/Users/uthpala/project-data/PyProcar/fermi-surface/MgB2-bands/abinit.out"
# kpoints_file = (
#     "/Users/uthpala/project-data/PyProcar/fermi-surface/MgB2-bands/kpath/KPOINTS"
# )

# NiO
procar_file = "/Users/uthpala/project-data/abinit/NiO/PROCAR"
abinit_file = "/Users/uthpala/project-data/abinit/NiO/abinit.out"
kpoints_file = "/Users/uthpala/project-data/abinit/NiO/KPOINTS"

abinit_object = AbinitParser(abinit_output=abinit_file)
recLat = abinit_object.reclat

procarFile = ProcarParser()
procarFile.readFile(procar_file)


pyprocar.bandsplot(
    procar_file,
    abinit_output=abinit_file,
    kpointsfile=kpoints_file,
    code="abinit",
    mode="plain",
    elimit=[-20, 20],
)


procar_file = "/Users/uthpala/project-data/PyProcar/fermi-surface/abinit/MgB2-G-centered-shifted/PROCAR"
abinit_file = "/Users/uthpala/project-data/PyProcar/fermi-surface/abinit/MgB2-G-centered-shifted/abinit.out"

abinit_object = AbinitParser(abinit_output=abinit_file)
recLat = abinit_object.reclat

procarFile = ProcarParser()
procarFile.readFile(procar_file)

pyprocar.fermi3D(
    procar_file,
    abinit_output=abinit_file,
    code="abinit",
    mode="plain",
    outcar=None,
    interpolation_factor=4,
    fermi_shift=-1.0,
)

# VASP
procar_file = "/Users/uthpala/project-data/PyProcar/fermi-surface/vasp/MgB2/PROCAR"
outcar = "/Users/uthpala/project-data/PyProcar/fermi-surface/vasp/MgB2/OUTCAR"

procarFile = ProcarParser()
procarFile.readFile(procar_file)


pyprocar.fermi3D(
    procar_file,
    mode="plain",
    code="vasp",
    outcar=outcar,
    interpolation_factor=4,
    # fermi_shift=-0.5,
)
