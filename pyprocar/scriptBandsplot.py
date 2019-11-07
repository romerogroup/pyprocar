  
from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect
from .procarplot import ProcarPlot
import numpy as np
import matplotlib.pyplot as plt
import re



  
def bandsplot(file,mode='scatter',color='blue',abinit_output=None,spin=0,atoms=None,orbitals=None,fermi=None,elimit=None,mask=None,markersize=0.02,cmap='jet',vmax=None,vmin=None,grid=True,marker='o',permissive=False,human=False,savefig=None,kticks=None,knames=None,title=None,outcar=None,kpointsfile=None):
  """This function plots band structures
  """
  #First handling the options, to get feedback to the user and check
  #that the input makes sense.
  #It is quite long
  if atoms is None:
    atoms = [-1]
    if human is True:
      print("WARNING: `--human` option given without atoms list!")
      print("--human will be set to False (ignored)\n ")
      human = False
  if orbitals is None:
    orbitals = [-1]
    

  print("Script initiated")
  print("input file    : ", file)
  print("Mode          : ", mode)
  
  print("spin comp.    : ", spin)
  print("atoms list.   : ", atoms)
  print("orbs. list.   : ", orbitals)

  if fermi is None and outcar is None and abinit_output is None:
    print("WARNING: Fermi Energy not set! ")
    print("You should use '-f' or '--outcar'\n Are you using Abinit Procar?\n")
    print("The zero of energy is arbitrary\n")
    fermi = 0

  if kpointsfile is None:
    print("No KPOINTS file present. Please set knames and kticks manually.")  


###################reading abinit output (added by uthpala) ##########################

  if abinit_output:
  	print("Abinit output used")

  #reading abinit output file
  	rf = open(abinit_output,'r')
  	data = rf.read()
  	rf.close()

  	fermi = float(re.findall('Fermi\w*.\(\w*.HOMO\)\s*\w*\s*\(\w*\)\s*\=\s*([0-9.+-]*)',data)[0])


####################################################################  

 
  print("Fermi Ener.   : ", fermi)
  print("Energy range  : ", elimit)

  if mask is not None:
    print("masking thres.: ", mask)
    
  print("Colormap      : ", cmap)
  print("MarkerSize    : ", markersize)
    
  print("Permissive    : ", permissive)
  if permissive:
    print("INFO: Permissive flag is on! Be careful")
  print("vmax          : ", vmax)
  print("vmin          : ", vmin)
  print("grid enabled  : ", grid) 
  if human is not None:
    print("human         : ", human)
  print("Savefig       : ", savefig)
  if kpointsfile is None:
    print("kticks        : ", kticks)
    print("knames        : ", knames)
  print("title         : ", title)

  print("outcar        : ", outcar)


  #If KPOINTS file is given:
  if kpointsfile is not None:
    #Getting the high symmetry point names from KPOINTS file
    f = open(kpointsfile)
    KPread = f.read()
    f.close()

    KPmatrix = re.findall('reciprocal[\s\S]*',KPread)
    tick_labels = np.array(re.findall('!\s(.*)',KPmatrix[0]))
    knames=[]
    knames=[tick_labels[0]]

    for i in range(len(tick_labels)-1):
      if tick_labels[i] !=tick_labels[i+1]:
        knames.append(tick_labels[i+1])

    knames = [str("$"+latx+"$") for latx in knames] 

    #getting the number of grid points from the KPOINTS file
    f2 = open(kpointsfile)
    KPreadlines = f2.readlines()
    f2.close()
    numgridpoints = int(KPreadlines[1].split()[0])

    kticks=[0]
    gridpoint=0
    for kt in range(len(knames)-1):
      gridpoint=gridpoint+numgridpoints
      kticks.append(gridpoint-1)
    print("knames        : ", knames)
    print("kticks        : ", kticks)  


  #If ticks and names are given by user manually:
  if kticks is not None and knames is not None:
    ticks = list(zip(kticks,knames))
  elif kticks is not None:
    ticks = list(zip(kticks,kticks))
  else:
    ticks = None


  
  #The spin argument should be a number (index of an array), or
  #'st'. In the last case it will be handled separately (later)

  spin = {'0':0, '1':1, '2':2, '3':3, 'st':'st'}[str(spin)]  


  #The second part of this function is parse/select/use the data in
  #OUTCAR (if given) and PROCAR

  #first parse the outcar if given, to get Efermi and Reciprocal lattice
  recLat = None 
  if outcar:
    outcarparser = UtilsProcar()
    if fermi is None:
      fermi = outcarparser.FermiOutcar(outcar)
      #if quiet is False:
      print("INFO: Fermi energy found in outcar file = " + str(fermi))
    recLat = outcarparser.RecLatOutcar(outcar)

  # parsing the PROCAR file
  procarFile = ProcarParser()
  procarFile.readFile(file, permissive, recLat)

  # processing the data, getting an instance of the class that reduces the data
  data = ProcarSelect(procarFile, deepCopy=True)
  
  #handling the spin, `spin='st'` is not straightforward, needs
  #to calculate the k vector and its normal. Other `spin` values
  #are trivial.
  if spin is 'st':
    #two `ProcarSelect` instances, to store temporal values: spin_x, spin_y
    dataX = ProcarSelect(procarFile, deepCopy=True)
    dataX.selectIspin([1])
    dataX.selectAtoms(atoms, fortran=human)
    dataX.selectOrbital(orbitals)  
    dataY = ProcarSelect(procarFile, deepCopy=True)
    dataY.selectIspin([2])
    dataY.selectAtoms(atoms, fortran=human)
    dataY.selectOrbital(orbitals)
    #getting the signed angle of each K-vector
    angle = np.arctan2(dataX.kpoints[:,1], (dataX.kpoints[:,0]+0.000000001))
    sin = np.sin(angle)
    cos = np.cos(angle)
    sin.shape = (sin.shape[0],1)
    cos.shape = (cos.shape[0],1)
    ##print sin, cos
    #storing the spin projection into the original array
    data.spd = -sin*dataX.spd + cos*dataY.spd
  else:
    data.selectIspin([spin])
    data.selectAtoms(atoms, fortran=human)
    data.selectOrbital(orbitals)

  # Plotting the data
  data.bands = (data.bands.transpose() - np.array(fermi)).transpose()
  plot = ProcarPlot(data.bands, data.spd, data.kpoints)

  ###### start of mode dependent options #########

  if mode == "scatter":
    plot.scatterPlot(mask=mask, size=markersize,
                     cmap=cmap, vmin=vmin,
                     vmax=vmax, marker=marker, ticks=ticks)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit is not None:
      plt.ylim(elimit)

  elif mode == "plain":
    plot.plotBands(markersize, marker=marker, ticks=ticks,color=color)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit:
      plt.ylim(elimit)
      
  elif mode == "parametric":
    plot.parametricPlot(cmap=cmap, vmin=vmin, vmax=vmax,
                        ticks=ticks)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit is not None:
      plt.ylim(elimit)

  elif mode == "atomic":
    plot.atomicPlot(cmap=cmap, vmin=vmin, vmax=vmax)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit is not None:
      plt.ylim(elimit)
  ###### end of mode dependent options ###########
  plt.tight_layout()
  if grid:
    plt.grid()
  
  if title:
    plt.title(title,fontsize=22)

  if savefig:
    plt.savefig(savefig,bbox_inches='tight')
  else:
    plt.show()

  return