from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect
from .procarplot import ProcarPlot
from .procarsymmetry import ProcarSymmetry
from .fermisurface import FermiSurface
import matplotlib.pyplot as plt



def fermi2D(file,outcar,spin=0,atoms=None,orbitals=None,energy=None,fermi=None,rec_basis=None,rot_symm=1,translate=[0,0,0],rotation=[0,0,0,1],human=False,mask=None,savefig=None,st=False,noarrow=False):
  """
  This module plots 2D Fermi surface.
  """
  if atoms is None:
    atoms = [-1]
    if human is True:
      print("WARNING: `--human` option given without atoms list!!!!!")

  if orbitals is None:
    orbitals = [-1]

  if rec_basis != None:
    rec_basis = np.array(rec_basis)
    rec_basis.shape = (3,3)

  if len(translate) != 3 and len(translate) != 1:
    print("Error: --translate option is invalid! (", translate,")")
    raise RuntimeError("invalid option --translate")

  
  print("file            : ", file)
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
  print("outcar          : ", outcar)
  print("st              : ", st)
  print("no_arrows       : ", noarrow)
  


  #first parse the outcar, if given
  if rec_basis is None and outcar:
    outcarparser = UtilsProcar()
    if fermi is None:
      fermi = outcarparser.FermiOutcar(outcar)
      print("Fermi energy found in outcar file = " + str(fermi))
    rec_basis = outcarparser.RecLatOutcar(outcar)
  #Reciprocal lattices are needed!
  elif rec_basis is None and outcar is None:
    print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
    raise RuntimeError("Reciprocal Lattice not found")
    
  #parsing the file
  procarFile = ProcarParser()
  #permissive incompatible with Fermi surfaces
  procarFile.readFile(file, permissive=False, recLattice=rec_basis)

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
    for i in [1,2,3]:
      data = ProcarSelect(procarFile)
      data.selectIspin([i])
      data.selectAtoms(atoms, fortran=human)
      data.selectOrbital(orbitals)
      stData.append(data.spd)

  #Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
  #symmetry operations to unfold the Brillouin Zone
  kpoints = data.kpoints
  bands = data.bands
  character = data.spd
  if st is True:
    sx, sy, sz = stData[0], stData[1], stData[2]
    symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz,character=character)
  else:
    symm = ProcarSymmetry(kpoints, bands, character=character)

  symm.Translate(translate)
  symm.GeneralRotation(rotation[0], rotation[1:])
  #symm.MirrorX()
  symm.RotSymmetryZ(rot_symm)


  # plotting the data
  print("Bands will be shifted by the Fermi energy = ", fermi)
  fs = FermiSurface(symm.kpoints, symm.bands-fermi, symm.character)
  fs.FindEnergy(energy)
  
  if not st:
    fs.Plot(mask=mask, interpolation=300)
  else:
    fs.st(sx=symm.sx, sy=symm.sy, sz=symm.sz, noarrow=noarrow, spin=spin)
  
  if savefig:
    plt.savefig(savefig)
  else:
    plt.show()
    
  return