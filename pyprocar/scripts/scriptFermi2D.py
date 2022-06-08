from ..utilsprocar import UtilsProcar
from ..io import ProcarParser
from ..procarselect import ProcarSelect
from ..procarplot import ProcarPlot
from ..procarsymmetry import ProcarSymmetry
from ..fermisurface import FermiSurface
from ..io import ElkParser
from ..io import AbinitParser
import matplotlib.pyplot as plt
from ..splash import welcome


def fermi2D(
    file,
    outcar=None,
    abinit_output=None,
    spin=0,
    atoms=None,
    orbitals=None,
    energy=None,
    fermi=None,
    rec_basis=None,
    rot_symm=1,
    translate=[0, 0, 0],
    rotation=[0, 0, 0, 1],
    human=False,
    mask=None,
    savefig=None,
    st=False,
    noarrow=False,
    exportplt=False,
    code="vasp",
    repair=True,
):
    """
  This module plots 2D Fermi surface.
  """

    welcome()

    # Turn interactive plotting off
    plt.ioff()

    # Repair PROCAR
    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(file, file)
            print("PROCAR repaired. Run with repair=False next time.")

    if atoms is None:
        atoms = [-1]
        if human is True:
            print("WARNING: `--human` option given without atoms list!!!!!")

    if orbitals is None:
        orbitals = [-1]

    if rec_basis != None:
        rec_basis = np.array(rec_basis)
        rec_basis.shape = (3, 3)

    if len(translate) != 3 and len(translate) != 1:
        print("Error: --translate option is invalid! (", translate, ")")
        raise RuntimeError("invalid option --translate")

    print("file            : ", file)
    print("outcar          : ", outcar)
    print("Abinit output   : ", abinit_output)
    print("atoms           : ", atoms)
    print("orbitals        : ", orbitals)
    print("spin comp.      : ", spin)
    print("energy          : ", energy)
    print("fermi energy    : ", fermi)
    print("Rec. basis      : ", rec_basis)
    print("rot. symmetry   : ", rot_symm)
    print("origin (trasl.) : ", translate)
    print("rotation        : ", rotation)
    print("masking thres.  : ", mask)
    print("save figure     : ", savefig)
    print("st              : ", st)
    print("no_arrows       : ", noarrow)

    # first parse the outputs if given
    if code == "vasp":
        if rec_basis is None and outcar:
            outcarparser = UtilsProcar()
            if fermi is None:
                fermi = outcarparser.FermiOutcar(outcar)
                print("Fermi energy found in outcar file = " + str(fermi))
            rec_basis = outcarparser.RecLatOutcar(outcar)
        # Reciprocal lattices are needed!
        elif rec_basis is None and outcar is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")

        # parsing the file
        procarFile = ProcarParser()
        # permissive incompatible with Fermi surfaces
        procarFile.readFile(file, permissive=False, recLattice=rec_basis)

    elif code == "elk":
        procarFile = ElkParser()
        if rec_basis is None:
            if fermi is None:
                fermi = procarFile.fermi
                print("Fermi energy found in Elk output file = " + str(fermi))
            rec_basis = procarFile.reclat
        # Reciprocal lattices are needed!
        if rec_basis is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")
        procarFile = Elkparser(kdirect=False)

    elif code == "abinit":
        if rec_basis is None and abinit_output:
            abinitparser = AbinitParser(abinit_output=abinit_output)
            if fermi is None:
                fermi = abinitparser.fermi
                print("Fermi energy found in Abinit ouput file = " + str(fermi))
            rec_basis = abinitparser.reclat
        # Reciprocal lattices are needed!
        elif rec_basis is None and abinit_output is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")
        # parsing the file
        procarFile = ProcarParser()
        # permissive incompatible with Fermi surfaces
        procarFile.readFile(file, permissive=False, recLattice=rec_basis)

    ### End of parsing ###

    if st is not True:
        # processing the data
        data = ProcarSelect(procarFile)
        data.selectIspin([spin])
        # fortran flag is equivalent to human,
        # but the later seems more human-friendly
        data.selectAtoms(atoms, fortran=human)
        data.selectOrbital(orbitals)
    else:
        # first get the sdp reduced array for all spin components.
        stData = []
        for i in [1, 2, 3]:
            data = ProcarSelect(procarFile)
            data.selectIspin([i])
            data.selectAtoms(atoms, fortran=human)
            data.selectOrbital(orbitals)
            stData.append(data.spd)

    # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
    # symmetry operations to unfold the Brillouin Zone
    kpoints = data.kpoints
    bands = data.bands
    character = data.spd
    if st is True:
        sx, sy, sz = stData[0], stData[1], stData[2]
        symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz, character=character)
    else:
        symm = ProcarSymmetry(kpoints, bands, character=character)

    symm.Translate(translate)
    symm.GeneralRotation(rotation[0], rotation[1:])
    # symm.MirrorX()
    symm.RotSymmetryZ(rot_symm)

    # plotting the data
    print("Bands will be shifted by the Fermi energy = ", fermi)
    fs = FermiSurface(symm.kpoints, symm.bands - fermi, symm.character)
    fs.FindEnergy(energy)

    if not st:
        fs.Plot(mask=mask, interpolation=300)
    else:
        fs.st(sx=symm.sx, sy=symm.sy, sz=symm.sz, noarrow=noarrow, spin=spin)

    if exportplt:
        return plt

    else:
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
            plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
        else:
            plt.show()
        return




# from typing import List

# import numpy as np
# import matplotlib.pyplot as plt

# from ..utilsprocar import UtilsProcar
# from ..io import ProcarParser
# from ..procarselect import ProcarSelect
# from ..procarplot import ProcarPlot
# from ..procarsymmetry import ProcarSymmetry
# from ..fermisurface import FermiSurface
# from ..io import ElkParser
# from ..io import AbinitParser
# from ..splash import welcome

# from .. import io


# def fermi2D(
#     file=None,
#     code:str="vasp",
#     outcar:str='OUTCAR',
#     poscar:str='POSCAR',
#     procar:str='PROCAR',
#     lobster:bool=False,
#     dirname:str='',
#     abinit_output=None,
#     spins:List[int]=[0],
#     atoms:List[int]=None,
#     orbitals:List[int]=None,
#     energy=None,
#     fermi:float=None,
#     rec_basis=None,
#     rot_symm=1,
#     translate:List[int]=[0, 0, 0],
#     rotation:List[int]=[0, 0, 0, 1],
#     human:bool=False,
#     mask=None,
#     savefig=None,
#     st:bool=False,
#     noarrow:bool=False,
#     exportplt:bool=False,
    
#     repair:bool=True,
# ):
#     """
#   This module plots 2D Fermi surface.
#   """

#     welcome()

#     # Turn interactive plotting off
#     plt.ioff()

#     # Repair PROCAR
#     if code == "vasp" or code == "abinit":
#         if repair:
#             repairhandle = UtilsProcar()
#             repairhandle.ProcarRepair(file, file)
#             print("PROCAR repaired. Run with repair=False next time.")

#     if atoms is None:
#         atoms = [-1]
#         if human is True:
#             print("WARNING: `--human` option given without atoms list!!!!!")

#     if orbitals is None:
#         orbitals = [-1]

#     if rec_basis != None:
#         rec_basis = np.array(rec_basis)
#         rec_basis.shape = (3, 3)

#     if len(translate) != 3 and len(translate) != 1:
#         print("Error: --translate option is invalid! (", translate, ")")
#         raise RuntimeError("invalid option --translate")

#     print("file            : ", file)
#     print("outcar          : ", outcar)
#     print("Abinit output   : ", abinit_output)
#     print("atoms           : ", atoms)
#     print("orbitals        : ", orbitals)
#     print("spin comp.      : ", spins)
#     print("energy          : ", energy)
#     print("fermi energy    : ", fermi)
#     print("Rec. basis      : ", rec_basis)
#     print("rot. symmetry   : ", rot_symm)
#     print("origin (trasl.) : ", translate)
#     print("rotation        : ", rotation)
#     print("masking thres.  : ", mask)
#     print("save figure     : ", savefig)
#     print("st              : ", st)
#     print("no_arrows       : ", noarrow)

#     # first parse the outputs if given
#     # if code == "vasp":
#     #     if rec_basis is None and outcar:
#     #         outcarparser = UtilsProcar()
#     #         if fermi is None:
#     #             fermi = outcarparser.FermiOutcar(outcar)
#     #             print("Fermi energy found in outcar file = " + str(fermi))
#     #         rec_basis = outcarparser.RecLatOutcar(outcar)
#     #     # Reciprocal lattices are needed!
#     #     elif rec_basis is None and outcar is None:
#     #         print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
#     #         raise RuntimeError("Reciprocal Lattice not found")

#     #     # parsing the file
#     #     procarFile = ProcarParser()
#     #     # permissive incompatible with Fermi surfaces
#     #     procarFile.readFile(file, permissive=False, recLattice=rec_basis)

#     # elif code == "elk":
#     #     procarFile = ElkParser()
#     #     if rec_basis is None:
#     #         if fermi is None:
#     #             fermi = procarFile.fermi
#     #             print("Fermi energy found in Elk output file = " + str(fermi))
#     #         rec_basis = procarFile.reclat
#     #     # Reciprocal lattices are needed!
#     #     if rec_basis is None:
#     #         print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
#     #         raise RuntimeError("Reciprocal Lattice not found")
#     #     procarFile = Elkparser(kdirect=False)

#     # elif code == "abinit":
#     #     if rec_basis is None and abinit_output:
#     #         abinitparser = AbinitParser(abinit_output=abinit_output)
#     #         if fermi is None:
#     #             fermi = abinitparser.fermi
#     #             print("Fermi energy found in Abinit ouput file = " + str(fermi))
#     #         rec_basis = abinitparser.reclat
#     #     # Reciprocal lattices are needed!
#     #     elif rec_basis is None and abinit_output is None:
#     #         print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
#     #         raise RuntimeError("Reciprocal Lattice not found")
#     #     # parsing the file
#     #     procarFile = ProcarParser()
#     #     # permissive incompatible with Fermi surfaces
#     #     procarFile.readFile(file, permissive=False, recLattice=rec_basis)

    

#     parser, reciprocal_lattice, e_fermi = parse(code=code,
#                                             lobster=lobster,
#                                             repair=repair,
#                                             dirname=dirname,
#                                             outcar=outcar,
#                                             poscar=poscar,
#                                             procar=procar,

#                                             fermi=fermi)

#     ### End of parsing ###

#     if st is not True:
#         # processing the data
#         # data = ProcarSelect(procarFile)
#         projected = parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
#         projected = projected[:,:,spins[0]]
#         # data.selectIspin([spin])
#         # fortran flag is equivalent to human,
#         # but the later seems more human-friendly
#         # data.selectAtoms(atoms, fortran=human)
#         # data.selectOrbital(orbitals)
#     # else:
#     #     # first get the sdp reduced array for all spin components.
#     #     stData = []
#     #     for i in [1, 2, 3]:
#     #         data = ProcarSelect(procarFile)
#     #         data.selectIspin([i])
#     #         data.selectAtoms(atoms, fortran=human)
#     #         data.selectOrbital(orbitals)
#     #         stData.append(data.spd)

#         # ebsX = copy.deepcopy(self.parser.ebs)
#         # ebsY = copy.deepcopy(self.parser.ebs)
#         # ebsZ = copy.deepcopy(self.parser.ebs)

#         # ebsX.projected = ebsX.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
#         # ebsY.projected = ebsY.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
#         # ebsZ.projected = ebsZ.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

#     # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
#     # symmetry operations to unfold the Brillouin Zone
#     # kpoints = data.kpoints
#     # bands = data.bands
#     # character = data.spd
#     kpoints = parser.ebs.kpoints
#     bands = parser.ebs.bands
#     character = projected
#     # if st is True:
#     #     sx, sy, sz = stData[0], stData[1], stData[2]
#     #     symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz, character=character)
#     # else:
#     symm = ProcarSymmetry(kpoints, bands, character=character)
#     symm.Translate(translate)
#     symm.GeneralRotation(rotation[0], rotation[1:])
#     # symm.MirrorX()
#     symm.RotSymmetryZ(rot_symm)

#     # plotting the data
#     # print("Bands will be shifted by the Fermi energy = ", fermi)
#     fs = FermiSurface(symm.kpoints, symm.bands, symm.character)
#     fs.FindEnergy(energy)

#     if not st:
#         fs.Plot(mask=mask, interpolation=300)
#     else:
#         fs.st(sx=symm.sx, sy=symm.sy, sz=symm.sz, noarrow=noarrow, spin=spins[0])

#     if exportplt:
#         return plt

#     else:
#         if savefig:
#             plt.savefig(savefig, bbox_inches="tight")
#             plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
#         else:
#             plt.show()
#         return


# def parse(code:str='vasp',
#           lobster:bool=False,
#           repair:bool=False,
#           dirname:str="",
#           outcar:str='OUTCAR',
#           poscar:str='PORCAR',
#           procar:str='PROCAR',
#           interpolation_factor:int=1,
#           fermi:float=None):
#         if code == "vasp" or code == "abinit":
#             if repair:
#                 repairhandle = UtilsProcar()
#                 repairhandle.ProcarRepair(procar, procar)
#                 print("PROCAR repaired. Run with repair=False next time.")

#         if code == "vasp":
#             outcar = io.vasp.Outcar(filename=outcar)
            
#             e_fermi = outcar.efermi
            
#             poscar = io.vasp.Poscar(filename=poscar)
#             structure = poscar.structure
#             reciprocal_lattice = poscar.structure.reciprocal_lattice

#             parser = io.vasp.Procar(filename=procar,
#                                     structure=structure,
#                                     reciprocal_lattice=reciprocal_lattice,
#                                     efermi=e_fermi,
#                                     )
#             # data = ProcarSelect(procarFile, deepCopy=True)


#         elif code == "qe":

#             if dirname is None:
#                 dirname = "bands"
#             parser = io.qe.QEParser(scfIn_filename = "scf.in", dirname = dirname, bandsIn_filename = "bands.in", 
#                                 pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
#                                 dos_interpolation_factor = None)
#             reciprocal_lattice = parser.reciprocal_lattice

#             e_fermi = parser.efermi


  

#         parser.ebs.bands += e_fermi
#         return parser, reciprocal_lattice, e_fermi