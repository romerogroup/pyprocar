#!/usr/bin/env python

from pyprocar.io.vasp import Procar, Outcar, Kpoints, Poscar


outcar_object = Outcar("vasp-procars/OUTCAR")
kpoints_object = Kpoints("vasp-procars/KPOINTS")
poscar_object = Poscar("vasp-procars/POSCAR")
procar_object = Procar(
    "vasp-procars/PROCAR",
    structure=poscar_object.structure,
    reciprocal_lattice=outcar_object.reciprocal_lattice,
    kpath=kpoints_object.kpath,
    efermi=outcar_object.efermi,
)

# plotting?
procar_object.ebs.plot()

# plot k-points
procar_object.ebs.plot_kpoints()
