  
from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect
from .procarplot import ProcarPlot
from .elkparser import ElkParser 
from .splash import welcome
import numpy as np
import matplotlib.pyplot as plt
import re

  
def bandsplot(file=None,mode='scatter',color='blue',abinit_output=None,spin=0,atoms=None,orbitals=None,fermi=None,elimit=None,mask=None,
            markersize=0.02,cmap='jet',vmax=None,vmin=None,grid=True,marker='o',permissive=False,human=False,savefig=None,kticks=None,
            knames=None,title=None,outcar=None,kpointsfile=None,exportplt=False, kdirect=True, discontinuities = [], code='vasp'):

  """This function plots band structures
  """
  welcome()

  # Turn interactive plotting off
  plt.ioff()

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
  print("code          : ", code)  
  print("input file    : ", file)
  print("Mode          : ", mode)  
  print("spin comp.    : ", spin)
  print("atoms list   : ", atoms)
  print("orbs. list   : ", orbitals)

  if fermi is None and outcar is None and abinit_output is None and code!='elk':
    print("WARNING: Fermi Energy not set! Please set manually or provide output file.")
    fermi = 0
  
  if fermi is None and code=='elk':
      fermi = None

  print("Fermi Energy   : ", fermi)
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
    if discontinuities:
        print("discontinuities :",discontinuities)
  print("title         : ", title)
  print("outcar        : ", outcar)

  if kdirect:
    print("k-points are in reduced coordinates")
  else:
    print("k-points are in cartesian coordinates. Remember to supply an output file for this case to work.")

  #### READING KPOINTS FILE IF PRESENT ####

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
   
    ################## Checking for discontinuities ########################
    discont_indx=[]
    icounter = 1
    while icounter<len(tick_labels)-1:
        if tick_labels[icounter] == tick_labels[icounter+1]:
            knames.append(tick_labels[icounter])
            icounter = icounter + 2
        else:
            discont_indx.append(icounter)
            knames.append(tick_labels[icounter]+'/'+tick_labels[icounter+1])
            icounter = icounter + 2
    knames.append(tick_labels[-1])                    
    discont_indx = list(dict.fromkeys(discont_indx))
    
    ################# End of discontinuity check ##########################
                  
    # Added by Nicholas Pike to modify the output of seekpath to allow for 
    # latex rendering.
    for i in range(len(knames)):
        if knames[i] =='GAMMA':
            knames[i] = '\Gamma'
        else:
            pass
            
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
    
    # creating an array for discontunuity k-points. These are the indexes 
    # of the discontinuity k-points.
    discontinuities = []  
    for k in discont_indx:
        discontinuities.append( kticks[int(k/2)+1]  )  
    if discontinuities:
        print("discontinuities :",discontinuities)  
        
  #### END OF KPOINTS FILE DEPENDENT SECTION ####      

   
  #The spin argument should be a number (index of an array), or
  #'st'. In the last case it will be handled separately (later)

  spin = {'0':0, '1':1, '2':2, '3':3, 'st':'st'}[str(spin)]  
  
  # parsing the PROCAR file
  if code == 'vasp' or code=='abinit':
      procarFile = ProcarParser()
  elif code == 'elk':
      procarFile = ElkParser(kdirect=kdirect)
      
      # Retrieving knames and kticks from Elk
      if kticks is None and knames is None:
          if procarFile.kticks and procarFile.knames:
              kticks = procarFile.kticks
              knames = procarFile.knames
      
  #If ticks and names are given by the user manually:
  if kticks is not None and knames is not None:
    ticks = list(zip(kticks,knames))
  elif kticks is not None:
    ticks = list(zip(kticks,kticks))
  else:
    ticks = None 
  

  #The second part of this function is parse/select/use the data in
  #OUTCAR (if given) and PROCAR

  #first parse the outcar if given, to get Efermi and Reciprocal lattice
  recLat = None     
  if code == 'vasp':    
      if outcar:
        outcarparser = UtilsProcar()
        if fermi is None:
          fermi = outcarparser.FermiOutcar(outcar)            
          print("Fermi energy found in outcar file = " + str(fermi))
        recLat = outcarparser.RecLatOutcar(outcar)
  
  elif code =='elk':
      if fermi is None:
        fermi = procarFile.fermi
        print("Fermi energy found in Elk output file = " + str(fermi))
  
  elif code=='abinit':  	
    if fermi is None:
      rf = open(abinit_output,'r')
      data = rf.read()
      rf.close()
      fermi = float(re.findall('Fermi\w*.\(\w*.HOMO\)\s*\w*\s*\(\w*\)\s*\=\s*([0-9.+-]*)',data)[0])  
      print("Fermi energy found in Abinit output file = " + str(fermi))
          

  # if kdirect = False, then the k-points will be in cartesian coordinates. 
  # The output should be read to find the reciprocal lattice vectors to transform from direct to cartecian
  
  if code=='vasp' or code=='abinit':
      if kdirect:        
        procarFile.readFile(file, permissive)        
      else:    
        procarFile.readFile(file, permissive, recLattice=recLat)
    

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
                     vmax=vmax, marker=marker, ticks=ticks, discontinuities = discontinuities)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit is not None:
      plt.ylim(elimit)

  elif mode == "plain":
    plot.plotBands(markersize, marker=marker, ticks=ticks,color=color, discontinuities = discontinuities)
    if fermi is not None:
    	plt.ylabel(r"$E-E_f$ [eV]",fontsize=22)
    else:
    	plt.ylabel(r"Energy [eV]",fontsize=22)	
    if elimit:
      plt.ylim(elimit)
      
  elif mode == "parametric":
    plot.parametricPlot(cmap=cmap, vmin=vmin, vmax=vmax,
                        ticks=ticks,discontinuities = discontinuities)
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

  if exportplt:
    return plt 

  else:
    if savefig:
      plt.savefig(savefig,bbox_inches='tight')
      plt.close() #Added by Nicholas Pike to close memory issue of looping and creating many figures
    else:
      plt.show()
    return  

#if __name__ == "__main__":
   # bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
   #knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
   #kticks=[0, 8, 16, 24, 38, 49])
    
