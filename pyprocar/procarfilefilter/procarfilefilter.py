import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys
from ..utilsprocar import UtilsProcar

class ProcarFileFilter:
  """Process a PROCAR file fields line-wise, specially useful for HUGE
  files. This could be thought as pre-processing, writting a new
  PROCAR-like file but reduced in some way.

  A PROCAR File is basically an multi-dimmensional arrays of data, the
  largest being:
  spd_data[#kpoints][#band][#ispin][#atom][#orbital]
  
  while the number of Kpoints d'ont seems a target for reduction
  (omission or averaging), the other fields can be reduced, for
  instance: grouping the atoms by species or as "surtrate" and
  "adsorbate", or just keeping the bands close to the Fermi energy, or
  discarding the d-orbitals in a s-p system. You got the idea, rigth?

  Example:

  -To group the "s", "p" y "d" orbitals from the file PROCAR and write
   them in PROCAR-spd:
   
   >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
   >>> a.FilterOrbitals([[0],[1,2,3],[4,5,6,7,8]], ['s','p', 'd'])
   
       The PROCAR-new will have just 3+1 columns (orbitals are colum-wise
       , take a look to the file). If you omit the ['s', 'p', 'd'] list, 
       the new orbitals will have a generic meaningless name (o1, o2, o3)

  -To group the atoms 1,2,5,6 and 3,4,7,8 from PROCAR and write them
   in PROCAR-new (note the 0-based indexes):

   >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
   >>> a.FilterAtoms([[0,1,4,5],[2,3,6,7]])

   -To select just the total density (ie: ignoring the spin-resolved stuff, 
    if any)from PROCAR and write it in PROCAR-new:

   >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
   >>> a.FilterSpin([0])

  """
  def __init__(self, infile=None, outfile=None,loglevel=logging.WARNING):
    """Initialize the class.

    Params: `infile=None`, input fileName
    """
    self.infile = infile
    self.outfile = outfile
        
    #We want a logging to tell us what is happening
    self.log = logging.getLogger("ProcarFileFilter")
    self.log.setLevel(loglevel)
    #This is a handler for logging, by now just keep it
    #untouched. Dont really matters its usage
    self.ch = logging.StreamHandler()
    self.ch.setFormatter(logging.Formatter("%(name)s::%(levelname)s:"
                                           " %(message)s"))
    self.ch.setLevel(logging.DEBUG)
    self.log.addHandler(self.ch)
    #At last, one message to the logger.
    self.log.debug("ProcarFileFilter instanciated")
    return

##########################SCRIPTS#################################

  # def scriptFilter(self,inFile,outFile,atoms=None,orbitals=None,orbital_names=None,bands=None,spin=None,human_atoms=False):
  #   print "Input file  :", inFile
  #   print "Output file :", outFile

  #   print "atoms       :", atoms
  #   if atoms:
  #     print "human_atoms     :", human_atoms
  #   print "orbitals  :", orbitals
  #   if orbitals:
  #       print "orb. names  :", orbital_names
  #     print "bands       :", bands
  #     print "spins       :", spin

  #   #Access init class of ProcarFileFilter and pass two arguments
  #   FileFilter = ProcarFileFilter(inFile,outFile)


  #   #for atoms
  #   if atoms:
  #     print "Manipulating the atoms"
      
  #     if human_atoms:
  #       atoms = [[y-1 for y in x] for x in atoms]
  #       print "new atoms list :", atoms

  #     #Now just left to call the driver member
  #     FileFilter.FilterAtoms(atoms)
    
  #   #for orbitals
  #   elif orbitals:
  #     print "Manipulating the orbitals"
  #     #If orbitals orbital_names is None, it needs to be filled
  #     if orbital_names is None:
  #       orbital_names = ["o"+str(x) for x in range(len(orbitals))]
  #       print "New orbitals names (default): ", orbital_names
  #     #testing if makes sense
  #     if len(orbitals) != len(orbital_names):
  #       raise RuntimeError("length of orbitals and orbitals names do not match")
      
  #     FileFilter.FilterOrbitals(orbitals,orbital_names)

  #   #for bands  
  #   elif bands:
  #     print "Manipulating the bands"
      
  #     bmin = bands[0]
  #     bmax = bands[1]
  #     if bmax < bmin:
  #       bmax, bmin = bmin, bmax
  #       print "New bands limits: ", bmin, " to ", bmax

  #     FileFilter.FilterBands(bmin,bmax)
      
  #   #for spin
  #   elif spin:
  #     print "Manipulating the spin"

  #     FileFilter.FilterSpin(spin)

  #   return









##################################################################
  
  def setInFile(self, infile):
    """Sets a input file `infile`, it can contains the path to the file"""
    self.infile = infile
    self.log.info("Input File: " + infile)
    return
  
  def setOutFile(self, outfile):
    """Sets a output file `outfile`, it can contains the path to the file"""
    self.outfile = outfile
    self.log.info("Out File: " + outfile)
    return
  
  def FilterOrbitals(self, orbitals, orbitalsNames):
    """
    Reads the file already set by SetInFile() and writes a new
    file already set by SetOutFile(). The new file only has the
    selected/grouped orbitals.

    Args: 

    -orbitals: nested iterable with the orbitals indexes to be
      considered. For example: [[0],[2]] means select the first
      orbital ("s") and the second one ("pz").
      [[0],[1,2,3],[4,5,6,7,8]] is ["s", "p", "d"].

    -orbitalsNames: The name to be put in each new orbital field (of a
      orbital line). For example ["s","p","d"] is a good
      `orbitalsName` for the `orbitals`=[[0],[1,2,3],[4,5,6,7,8]]. 
      However, ["foo", "bar", "baz"] is equally valid.

    Note: 
      -The atom index is not counted as the first field.
      -The last column ('tot') is so important that it is always
       included. Do not needs to be called
    """
    # setting iostuff, this method -and class- should not made any
    # checking about IO, that is the job of the caller
    self.log.info("In File: " + self.infile)
    self.log.info("Out File: " + self.outfile)
    # open the files
    fout = open(self.outfile, 'w')
    fopener = UtilsProcar()
    fin = fopener.OpenFile(self.infile)
    for line in fin:
      if re.match(r"\s*ion\s*", line):
        #self.log.debug("orbital line found: " + line)
        line = " ".join(['ion'] + orbitalsNames + ['tot']) + "\n"
          
      elif re.match(r"\s*\d+\s*", line) or re.match(r"\s*tot\s*", line):
        #self.log.debug("data line found: " + line)
        line = line.split()
        #all floats to an array
        data = np.array(line[1:], dtype=float)
        #setting a new line, keeping just the first value
        line = line[:1]
        for orbset in orbitals:
          line.append(data[orbset].sum())
        #the last value ("tot") always  should be written
        line.append(data[-1])
        #converting to str
        line =  [str(x) for x in line] 
        line = " ".join(line) + "\n"
      fout.write(line)
        
    return
  
  def FilterAtoms(self, atomsGroups):
    """
    Reads the file already set by SetInFile() and writes a new
    file already set by SetOutFile(). The new file only has the
    selected/grouped atoms.

    Args: 

    -atomsGroups: nested iterable with the atoms indexes (0-based) to
      be considered. For example: [[0],[2]] means select the first and
      the second atoms. While [[1,2,3],[4,5,6,7,8]] means select the
      contribution of atoms 1+2+3 and 4+5+6+7+8

    Note: 
      -The atom index is c-based (or python) beginning with 0
      -The output has a dummy atom index, without any intrisic meaning
    
    """
    # setting iostuff, this method -and class- should not made any
    # checking about IO, that is the job of the caller
    self.log.info("In File: " + self.infile)
    self.log.info("Out File: " + self.outfile)
    # open the files
    fout = open(self.outfile, 'w')
    fopener = UtilsProcar()
    with fopener.OpenFile(self.infile) as fin:
      # I need to change the numbers of ions, it will needs the second
      # line. The first one is not needed
      fout.write(fin.readline())
      line = fin.readline()
      line = line.split()
      # the very last value needs to be changed
      line[-1] = str(len(atomsGroups))
      line = ' '.join(line)
      fout.write(line + '\n')
      
      # now parsing the rest of the file
      data = []
      for line in fin:
        # if line has data just capture it
        if re.match(r"\s*\d+\s*", line):
          # self.log.debug("atoms line found: " + line)
          data.append(line)
        # if `line` is a end of th block (begins with 'tot'), do the
        # work. And clean up data then
        elif re.match(r"\s*tot\s*", line):
          # self.log.debug("tot line found: " + line)
          # making an array
          data = [x.split() for x in data]
          data = np.array(data, dtype=float)
          # iterating on the atoms groups
          for index in range(len(atomsGroups)):
            atoms = atomsGroups[index]
            # summing colum-wise
            atomLine = data[atoms].sum(axis=0)
            atomLine = [str(x) for x in atomLine]
            # the atom index should not be averaged (anyway now is
            # meaningless)
            atomLine[0] = str(index+1)
            atomLine = ' '.join(atomLine) 
            fout.write(atomLine + '\n' )
          
          # clean the buffer
          data = []
          # and write the `tot` line
          fout.write(line)
        # otherwise just write this line
        else:
          fout.write(line)

    return

  def FilterBands(self, Min, Max):
    """
    Reads the file already set by SetInFile() and writes a new
    file already set by SetOutFile(). The new file only has the
    selected bands.

    Args: 

    -Min, Max:
      the minimum/maximum band  index to be considered, the indexes 
      are the same used by vasp (ie written in the file).


    Note: -Since bands are somewhat disordered in vasp you may like to
      consider a large region and made some trial and error
    
    """
    # setting iostuff, this method -and class- should not made any
    # checking about IO, that is the job of the caller
    self.log.info("In File: " + self.infile)
    self.log.info("Out File: " + self.outfile)
    # open the files
    fout = open(self.outfile, 'w')
    fopener = UtilsProcar()
    fin = fopener.OpenFile(self.infile)

    # I need to change the numbers of kpoints, it will needs the second
    # line. The first one is not needed
    fout.write(fin.readline())
    line = fin.readline()
    # the third value needs to be changed, however better print it
    self.log.debug("The line contaning bands number is " + line)
    line = line.split()
    self.log.debug("The number of bands is: " + line[7])
    line[7] = str(Max-Min+1)
    line = ' '.join(line)
    fout.write(line + '\n')
    
    # now parsing the rest of the file
    write = True
    for line in fin:
      if re.match(r"\s*band\s*", line):
        # self.log.debug("bands line found: " + line)
        band = int(re.match(r"\s*band\s*(\d+)", line).group(1)) 
        if band < Min or band > Max:
          write = False
        else:
          write = True
      if re.match(r"\s*k-point\s*", line):
        write = True
      if write:
        fout.write(line)
    return
  
  def FilterSpin(self, components):
    """Reads the file already set by SetInFile() and writes a new
    file already set by SetOutFile(). The new file only has the
    selected part of the density (sigma_i).

    Args: 

    -components: The spin component block, for instante [0] menas just
      the density, while [1,2] would be the the sigma_x and sigma_y
      for a non-collinear calculation.

    TODO: spin-polarized collinear case is not included at all, not
    even a warning message!

    """
    # setting iostuff, this method -and class- should not made any
    # checking about IO, that is the job of the caller
    self.log.info("In File: " + self.infile)
    self.log.info("Out File: " + self.outfile)
    # open the files
    fout = open(self.outfile, 'w')
    fopener = UtilsProcar()
    with fopener.OpenFile(self.infile) as fin:
      counter = 0
      for line in fin:
        # if any data found 
        if re.match(r"\s*\d", line):
          # check if should be written
          if counter in components:
            fout.write(line)
        elif re.match(r"\s*tot", line):
          if counter in components:
            fout.write(line)
          # the next block will belong to other component
          counter += 1
        elif re.match(r"\s*ion", line):
          fout.write(line)
          counter = 0
        else:
          fout.write(line)
    return

  def FilterKpoints(self, Min, Max):
    """
    Reads the file already set by SetInFile() and writes a new
    file already set by SetOutFile(). The new file the
    selected bands.

    Args: 

    -Min, Max:
      the minimum/maximum band  kpoint to be considered, the indexes 
      are the same used by vasp (i.e. written in the file). Not starting from zero
    """
    # setting iostuff, this method -and class- should not made any
    # checking about IO, that is the job of the caller
    self.log.info("In File: " + self.infile)
    self.log.info("Out File: " + self.outfile)
    # open the files
    fout = open(self.outfile, 'w')
    fopener = UtilsProcar()
    fin = fopener.OpenFile(self.infile)

    # I need to change the numbers of kpoints, it will needs the second
    # line. The first one is not needed
    fout.write(fin.readline())
    line = fin.readline()
    # the third value needs to be changed, however better print it
    self.log.debug("The line contaning kpoints number is " + line)
    line = line.split()
    self.log.debug("The number of kpoints is: " + line[3])
    line[3] = str(Max-Min+1)
    line = ' '.join(line)
    fout.write(line + '\n')
    
    # now parsing the rest of the file
    write = True
    for line in fin:
      if re.match(r"\s*k-point\s*", line):
        self.log.debug("bands line found: " + line)
        kpoint = int(re.match(r"\s*k-point\s*(\d+)", line).group(1)) 
        if kpoint < Min or kpoint > Max:
          write = False
        else:
          write = True
      if write:
        fout.write(line)
    return