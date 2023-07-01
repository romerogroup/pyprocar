#!/usr/bin/env python
import sys
import re
import numpy as np
import argparse


class Poscar:
  """Stores the settings of a POSCAR. The Cartesian and direct coords
    must be updated together all the time.  The scaling factor, is
    internally set to 1.0, always.

    Methods:
    parse(self)                    # loads the whole file
    _set_cartesian(self)           # set direct -> cartesian (internal)
    _set_direct(self)              # set cartesian -> direct (internal)
    _unparse(self, direct)         # data to string, called by write()
    write(self, filename, direct)  # saves the class to a POSCAR-like file
    xyz(self, filename)            # saves a xyz file from the data
    sort(self)                     # sorts the atoms, grouping them by element
    remove()                       # removes one or more atoms from poscar
    add(position, element, direct) # add one atom at the time
  """

  def __init__(self, filename, verbose=False):
    self.verbose = verbose
    self.filename = filename
    self.poscar = None
    self.cpos = None # cartesian coordinates
    self.dpos = None # direct coordinates
    self.lat = None # lattice
    self.typeSp = None # Name of atomic species
    self.numberSp = None # Number of atoms per specie
    self.Ntotal = None # Total atoms in system
    self.elm = None # Element of each atoms one-by-one. list(str)
    self.selective = None # Selective dynamics
    self.selectFlags = None # all the T,F from selective dynamics. np.array(str)
    self.volume = None
    return
    
  def parse(self, fromString=None):
    """Loads into memory all the content of the POSCAR file.
    Args:
    
    `fromString` : If present, instead of loading a file, it uses this
    variable as a string with the contents of the POSCAR file. Default=None

    """
    if fromString and isinstance(fromString, str):
      self.poscar = fromString.split('\n')
    elif fromString and isinstance(fromString, list):
      self.poscar = fromString
    else:
      self.poscar = open(self.filename, 'r')
      self.poscar = self.poscar.readlines()

    # getting the scale factor
    scale = re.findall(r'[.\d]+', self.poscar[1])
    scale = float(scale[0])
    if self.verbose:
      print('scaling factor: ', scale)
    
    # parsing the lattice
    self.lat = re.findall(r'[-.\d]+\s+[-.\d]+\s+[-.\d]+\s*\n*', ''.join(self.poscar[2:5]))
    self.lat = np.array([x.split() for x in self.lat], dtype=float)*scale
    if self.verbose:
      print( 'lattice:\n', self.lat)

    # type of atoms and number of atoms per type
    self.typeSp = re.findall(r'(\w+)\s*', self.poscar[5])
    if len(self.typeSp) == 0:
      raise RuntimeError("No data about the atomic species found. Correct it.")
    if self.verbose:
      print( 'atoms per species:\n', self.typeSp)
    self.numberSp = re.findall(r'(\d+)\s*', self.poscar[6])
    self.numberSp = np.array(self.numberSp, dtype=int)
    self.Ntotal = np.sum(self.numberSp)
    if self.verbose:
      print( 'atomic species:\n', self.numberSp)
      print( 'The total is ' + str(self.Ntotal) + ' atoms')
    
    # selective dynamics? setting a offset
    if re.findall(r'^\s*[sS]', self.poscar[7]) :
      self.selective = True
      offset = 1
      if self.verbose:
        print( "Selective Dynamics")
    else:
      self.selective = False
      offset = 0
      
    # Direct coordinates unless otherwise
    direct = True
    if not re.findall(r'^\s*[Dd]', self.poscar[7+offset]) :
      direct = False
      if self.verbose:
        print('Positions in Cartesian coordinates')
    else:
      if self.verbose:
        print('Positions in Direct coordinates')

    # parsing the positions
    start, end = 8+offset, 8+offset+ self.Ntotal
    string = r'([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*'
    pos = re.findall(string, ''.join(self.poscar[start:end]))
    if direct:
      self.dpos = np.array(pos, dtype=float)
      self._set_cartesian()
    else:
      self.cpos = np.array(pos, dtype=float)
      self._set_direct()
    if self.verbose:
      print( "Atomic positions (direct)\n", self.dpos)
      print( "Atomic positions (cartesian)\n", self.cpos)

    # parsing selective dynamics
    if self.selective == True:
      string = r'([TF]+)\s+([TF]+)\s+([TF]+)\s+'
      selectFlags = re.findall(string, ''.join(self.poscar[start:end]))
      self.selectFlags = np.array(selectFlags)
      print('Flags of selective dynamics:\n', self.selectFlags)
      
    # setting a list of elements:
    elementList = zip(self.typeSp, self.numberSp)
    self.elm = ' '.join([' '.join([x]*y) for x,y in elementList]).split()
    if self.verbose:
      print('Elements: ', self.elm)

    # setting the volume, just as an utility
    self.volume = np.linalg.det(self.lat)
    return

  def _set_cartesian(self):
    cart = np.dot(self.lat.T, self.dpos.T)
    cart = cart.T
    self.cpos = cart
    
  def _set_direct(self):
    inverse = np.linalg.inv(self.lat)
    direct = np.dot(inverse.T, self.cpos.T)
    direct = direct.T
    self.dpos = direct
    
  def _unparse(self, direct=True):
    """writes a POSCAR file with the info stored in the arrays. The
    information is as it is. No PBC are applied, and no checks are
    performed at this stage.
    The scaling factor is 1.0, always.
    """

    # We will start with getting the positions
    if direct == True:
      pos = self.dpos
    else:
      pos = self.cpos
      
    # creating a list of text lines with positions
    pos = [' '.join([str(coord) for coord in line]) for line in pos]

    # Now we will look whether selective dynamics are used
    if self.selective == True:
      # a list of text lines with flags
      flags = [' '.join([flag for flag in line]) for line in pos]
      pos = [pos + ' ' + flag for (pos, flag) in zip(pos,flags)]

    pos = '\n'.join(pos)

    # Creating the POSCAR text string
    self.poscar = "poscar.py\n"
    self.poscar += "1.0\n"
    self.poscar += '\n'.join([' '.join([str(y) for y in x]) for x in self.lat]) + '\n'
    self.poscar += ' '.join(self.typeSp) + '\n'
    self.poscar += ' '.join([str(x) for x in self.numberSp]) + '\n'
    if self.selective:
      self.poscar += 'Selective Dynamics\n'
    if direct == False:
      self.poscar += 'Cartesian\n'
    else:
      self.poscar += 'Direct\n'
    self.poscar += pos # already set with the the T, F -if needed
    self.poscar += '\n'
    
    if self.verbose:
      print("\n\n unparsed POSCAR\n\n")
      print('unparse, self.poscar\n', self.poscar)


  def write(self, filename='POSCAR.out', direct=True):
    self._unparse(direct=direct)
    fout = open(filename, 'w')
    fout.write(self.poscar)
    if self.verbose:
      print('File '  + filename + ' written.')
    return
      
  def xyz(self, filename):
    xyzf = open(filename, 'w')
    xyzf.write(str(self.Ntotal) + '\n')

    # The comment line has the lettice
    latStr = np.array(self.lat, dtype=str).flatten()
    latStr = ' '.join(latStr)
    xyzf.write('Lattice="'  + latStr + '"\n')
    
    # continuing with the positions
    pos = self.cpos
    pos = np.array(pos, dtype=str)
    pos = [' '.join(x) for x in pos]
    elm = list(self.elm)
    xyzstr = '\n'.join( [ x + ' ' + y for x,y in zip(elm, pos) ] )
    xyzf.write(xyzstr)
    xyzf.write('\n')
    xyzf.close()
    if self.verbose:
      print(filename + ' written as xyz')
      
  def sort(self):
    #
    # self.typeSp, self.elem, self.dpos
    # must be present and updated (they can be disordered)
    #
    from collections import OrderedDict
    # getting the different element's names, without repetitions
    self.typeSp = list(OrderedDict.fromkeys(self.typeSp))
    if self.verbose:
      print('The list of elements is ', self.typeSp)
    # lists of ordered atoms
    atoms = []
    elements = []
    if self.verbose:
      print('to sort: ', self.elm, '\n', self.dpos)
    
    # ordering the positions accordiong to its element.
    for thiselem in self.typeSp:
      # Joining each element with its position
      for elem, pos in zip(self.elm, self.dpos):
        if thiselem == elem:
          atoms.append(pos)
          elements.append(elem)
    if self.verbose:
      print('sorted: ', elements, '\n', np.array(atoms))

    self.elm = elements
    # setting the position's list in direct coords.
    self.dpos = np.array(atoms)
    # and in cartesian coords as well
    self._set_cartesian()

    # How many atoms by element?
    from collections import Counter
    # a dict of elements and its repetitions
    counter = Counter(elements)
    self.numberSp = [counter[x] for x in self.typeSp]
    
    self.Ntotal = sum(self.numberSp)
    if self.verbose:
      print ("N atoms per specie, ", self.numberSp, '. Total: ', self.Ntotal)

  def remove(self, atoms):
    # atoms maybe (or not) just one atom (an int, not a one-sized list)
    if self.verbose:
      print('going to delete the following atom(s):', atoms)
    try:
      iterator = iter(atoms)
    except TypeError:
      atoms = [atoms]
      
    # we will populate a list with the atoms to keep
    keep = [True]*self.Ntotal
    for atom in atoms:
      if atom >= self.Ntotal:
        raise RuntimeError('Error: atom index is larger than the atom number')
      keep[atom] = False
    # creating new arrays without the removed elements
    self.cpos = self.cpos[keep]
    self.dpos = self.dpos[keep]
    self.Ntotal = len(self.cpos)
    
    # self.elm is a list, I am not sure why, but I will cast it back to list
    self.elm = np.array(self.elm)
    self.elm = self.elm[keep]
    self.elm = list(self.elm)
    
    if self.selective:
      self.selectFlags = self.selectFlags[keep]
    # the atoms types and their number can be modified, we need to
    # count them from self.elm
    # Also I want to keep the order of elements
    from collections import OrderedDict
    self.typeSp = list(OrderedDict.fromkeys(self.elm).keys())
    self.numberSp = [self.elm.count(x) for x in self.typeSp]
    if self.verbose:
      print('Elements', self.elm)
      print(self.typeSp, self.numberSp)
    # no need to sort, deleting doesn't alters the occurence
    return

  def add(self, position, element, direct=True, selectiveFlags=None):
    """ Atoms must be added one at the time """
    position = np.array(position, dtype=float)
    position.shape = (1,3)
    if self.verbose:
      print('going to add an ' +element+  ' atom at', position, end=',')
      if direct:
        print('in direct coordinates')
      else:
        print('in Cartesian coordiantes')
    # setting the data
    if direct:
      self.dpos = np.concatenate((self.dpos, position))
    else:
      self.cpos = np.concatenate((self.cpos, position))
    self.Ntotal = self.Ntotal + 1
    self.elm.append(element)
    if self.selective and selectiveFlags:
      if self.verbose:
        print('selective flag found.')
      selectiveFlags = np.array(selectiveFlags, dtype=str)
      self.selectFlags = np.concatenate(self.selectFlags, selectiveFlags)
    # setting the other data
    if direct:
      self._set_cartesian()
    else:
      self._set_direct()
    # the atoms types and their number can be modified, we need to
    # count them from self.elm
    # Also I want to keep the order of elements
    from collections import OrderedDict
    self.typeSp = list(OrderedDict.fromkeys(self.elm).keys())
    self.numberSp = [self.elm.count(x) for x in self.typeSp]
    if self.verbose:
      print('Elements', self.elm)
      print(self.typeSp, self.numberSp)
    # sorting the data
    self.sort()
    return

    
        
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("inputfile", type=str, help="input file")
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('--xyz', action='store_true')
  
  args = parser.parse_args()

  p = Poscar(args.inputfile, verbose=args.verbose)
  p.parse()

  if args.xyz:
    p.xyz(args.inputfile + '.xyz')
