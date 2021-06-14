#!/usr/bin/env python

from pyprocar.io.abinit import AbinitParser
from pyprocar.io.abinit import AbinitKpoints

# Parsing Abinit PROCAR file and output file
ab_object = AbinitParser(abinit_output="abinit.out")
ab_procar_ob = ab_object.abinitprocarobject
ebs_object = ab_procar_ob.ebs

# Parsing k-point info from abinit output file
ab_kpoints = AbinitKpoints()
