import re
import argparse
import numpy as np

from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath

class DFTB_evec:
    def __init__(self, filename, verbose): # , normalize=False):
        self.verbose = verbose
        self.filename = filename
        # self.normalized = normalize # do I need to normalize the eigenvectors?
        self.f = None # the whole file
        self.Nkpoints = None # How many k-points
        self.Nbands = None # How many bands
        self.Natoms = None
        self.Norbs = None # How many orbitals (as a whole quantity, not by atom)
        
        # np.array[Nkpoints, Nbands, Natoms, Norbs], it can be complex or real.
        #The Mulliken part is ignored.
        self.spd = None 
        self.orbDict = None # An ordered dictionary assigning an index to each orbital
        # Is the data complex or real? Nkpoint=1->real, Nkpoints>1->complex
        self.is_complex = None
        # The eigenvalues (Nkpoints, Nbands) to be set externally, the
        # file with eigenvectors doesn't have this information
        self.bands = None
        # The kpoints (Nkpoints, 4), the last entry are the
        # weights. They are expected to be in direct
        # coordinates. Needs to be set externally.
        self.kpoints = None 
        self.occupancies = None # (Nkpoints, Nbands), to be set externally.
        self.phase = None
        return
    
    def load(self):
        self.f = open(self.filename, 'r')
        self.f = self.f.read()
        # The syntax is different whether there are more than one k-point
        # or just a single k-point.
        klist = re.findall(r'K-point\s*:\s*\d+', self.f)
        klist = set(klist) # using `set` to remove repeated elements
        self.Nkpoints = len(klist)
        # if no kpoints found, there is only one
        if self.Nkpoints == 0:
            self.Nkpoints = 1
        if self.verbose:
            print('Number of k-points: ', self.Nkpoints)
        if self.Nkpoints == 1:
            self.is_complex = False
        else:
            self.is_complex = True
        if self.verbose:
            print('Looking for complex data?', self.is_complex)

        # How many eigenvectors?
        band_list = re.findall(r'Eigenvector\s*:\s*\d+', self.f)
        up_list = re.findall(r'Eigenvector\s*:\s*\d+\s*\(up\)', self.f)
        if len(band_list) != len(up_list):
            print('Number of bands: ', len(band_list), ' Number of UP bands: ', len(up_list))
            raise RuntimeError('Spin polarization not supported')
        up_list = set(up_list)
        self.Nbands = len(up_list)
        if self.verbose:
            print('Number of bands: ', self.Nbands)

        # How many atoms?
        atoms_list = re.findall('^\s*\d+\s+\w+', self.f, re.MULTILINE)
        atoms_list = set(atoms_list)
        self.Natoms = len(atoms_list)
        if self.verbose:
            print('Number of atoms: ', len(atoms_list))
            # print(atoms_list)
      
        # How many orbitals?
        orb_list = re.findall(r'[ ]+([a-zA-Z]+[-_a-zA-z\d]*)[ ]+[.(\d\-]+', self.f)
        self.Norbs = len(set(orb_list))
        print(set(orb_list))
        if self.verbose:
            print('Number of orbitals: ', self.Norbs)

        # also, we are going to create a Dict of the unique
        # orbitals, assigning them an index to store them into the arrays.
        # We are faking an orderedSet by using an orderedDict
        from collections import OrderedDict
        d = OrderedDict.fromkeys(orb_list)
        i = 0
        for key in d.keys():
            d[key] = i
            i = i+1
        self.orbDict = d
        if self.verbose:
            print('Set of orbitals and their indexes: ', self.orbDict)
            
            
        # Creating a big array with all the info. It needs to be
        # initialized to zero, since not all fields are present in the
        # file
        if self.is_complex:
            self.spd = np.zeros([self.Nkpoints, self.Nbands, self.Natoms,
                                 self.Norbs], dtype=complex)
        else:
            self.spd = np.zeros([self.Nkpoints, self.Nbands, self.Natoms,
                                 self.Norbs], dtype=float)
      
        # splitting the file by Eigenvector (and perhaps K-point):
        #
        # The first match is 'Coefficients and Mulliken populations of the
        # atomic orbitals', and needs to be discarded
        
        if self.Nkpoints == 1:
            # Eigenvector:   1    (up)
            evecs = re.split(r'Eigenvector:\s*\d+\s*\(up\)\s*', self.f)[1:]
        else:
            # K-point:    1    Eigenvector:    1    (up)
            evecs = re.split(r'K-point:\s*\d+\s*Eigenvector:\s*\d+\s*\(up\)\s*', self.f)[1:]
        if self.verbose:
            print('We got :', len(evecs), 'Eigenvector entries (expected: ',
                  self.Nkpoints*self.Nbands, ')' )
        # temporal storage of the evecs as an Nkpoint*Nbands array
        evecs = np.array(evecs)
        evecs.shape = (self.Nkpoints, self.Nbands)

        # going to parse all the data, iterating over:
        # Kpoints->Bands->Atoms (and assignation at orbital level)

    
        for cKpoint in range(self.Nkpoints):
            for cEvec in range(self.Nbands):
                # just dealing with the data of the current block
                evec = evecs[cKpoint, cEvec]
                # splitting by atom
                evec = re.split(r'\n*\s*\d+\s*[a-zA-Z]+\s+', evec)
                # the first match should be '', and should be discarded
                if evec[0] == '':
                    evec.pop(0)
                #if self.verbose:
                #  print('number of atoms in the block',  len(evec))
            
                cAtom = 0 # a counter of atoms
                for atom in evec:
                    # next we search for the orbitals involved
                    orbNames = (re.findall(r'[a-zA-Z_]+[-_a-zA-z\d]*', atom))
                    # what is the index of each orbital? 
                    orbIndexes = [self.orbDict[x] for x in orbNames]
                    #print(orbNames, orbIndexes)
                    # getting all the floating numbers
                    numbers = re.findall('[\-0-9]+\.\d+', atom)
                    numbers = np.array(numbers, dtype=float)
                    # if they are complex, we need to cast them. The last column
                    # nneds to be ignored.
                    if self.is_complex:
                        numbers.shape = (len(orbIndexes), 3) # Re, Im, Mulliken
                        numbers = numbers[:,0] + 1j*numbers[:,1]
                    else:
                        numbers.shape = (len(orbIndexes), 2) # Re. Mulliken
                        numbers = numbers[:,0]
                    # print(numbers)
                    self.spd[cKpoint, cEvec, cAtom, orbIndexes] = numbers
                    cAtom = cAtom + 1
        # setting the phases
        # setting the projections as the square of wave coefficients
        if self.is_complex:
            self.phase = np.angle(self.spd)
            self.spd = np.array((self.spd * np.conjugate(self.spd)).real, dtype=float)           
        else:
            self.phase = np.zeros(self.spd.shape)
            self.spd = np.array(self.spd**2, dtype=float)
        if self.verbose:
            print('file', self.filename, 'loaded')
    
        print('Overlap Populations (Mulliken) are ignored')
        # if self.normalized:
        #   if self.verbose:
        #     print('The projection of eigenvectors will be normalized')
        #   #norms = np.zeros((self.Nkpoints, self.Nbands), dtype=float)
        #   for k in range(self.Nkpoints):
        #     for b in range(self.Nbands):
        #       self.spd[k,b] /= np.linalg.norm(self.spd[k,b])
        #       #norms[k,b] = np.linalg.norm(self.spd[k,b])
        #     #self.spd = self.spd/norms
        return
    
    def set_bands(self, bands, occupancies):
        if self.verbose:
            print('Setting the bands into the eigenvectors class')
        if bands.shape == (self.Nkpoints, self.Nbands):
            self.bands = bands
        else:
            raise RuntimeError('The number of the bands array doesn\'t'
                               'match: ' + str(bands.shape) + ' vs ' +
                               str((self.Nkpoints, self.Nbands)))
        if occupancies.shape == (self.Nkpoints, self.Nbands):
            self.occupancies = occupancies
        else:
            raise RuntimeError('The number of the occupancies array doesn\'t'
                               'match: ' + str(occupancies.shape) + ' vs ' +
                               str((self.Nkpoints, self.Nbands)))
        return
  
    def set_kpoints(self, kpoints):
        if self.verbose:
            print('Setting the kpoints into the eigenvectors class')
        if kpoints.shape == (self.Nkpoints, 4):
            self.kpoints = kpoints
        else:
            raise RuntimeError('The shape of the kpoints array doesn\'t'
                               'match: ' + str(kpoints.shape) + ' vs ' +
                               str((self.Nkpoints, 4)))
        return

    def writeProcar(self):
        # I need to sum over orbitals, and over atoms and over both
        tot_orbs = np.sum(self.spd, axis=3)
        tot_atoms = np.sum(self.spd, axis=2)
        tot_oa = np.sum(tot_orbs, axis=2) # already summed over axis=3
        print('sum over orbitals.shape: ', tot_orbs.shape)
        print('sum over atoms.shape: ', tot_atoms.shape)
    
        # casting to strings
        if self.verbose:
            print('going to transform all data to strings: ', end='')
          
        self.spd = np.array(["%.5f" % x for x in self.spd.flatten()])
        self.spd.shape = (self.Nkpoints, self.Nbands, self.Natoms, self.Norbs)
    
        tot_orbs = np.array( ["%.5f" % x for x in tot_orbs.flatten()] )
        tot_orbs.shape = (self.Nkpoints, self.Nbands, self.Natoms)
    
        tot_atoms = np.array( ["%.5f" % x for x in tot_atoms.flatten()] )
        tot_atoms.shape = (self.Nkpoints, self.Nbands, self.Norbs)
    
        tot_oa = np.array( ["%.5f" % x for x in tot_oa.flatten()] )
        tot_oa.shape = (self.Nkpoints, self.Nbands)
        
        # also the energies need to be casted to strings safely (i.e. no
        # scientific notation)
        bands = np.array( ["%.6f" % x for x in self.bands.flatten()] )
        bands.shape = (self.Nkpoints, self.Nbands)
    
        # casting kpoints and weigths 
        kpoints = np.array( ["%.6f" % x for x in self.kpoints.flatten()] )
        kpoints.shape = (self.Nkpoints, 4)
    
        # and occupancies
        occupancies = np.array( ["%.5f" % x for x in self.occupancies.flatten()] )
        occupancies.shape = (self.Nkpoints, self.Nbands)
        
        if self.verbose:
            print('done')
    
        # writing the data
          
        f = open('PROCAR', 'w')
        # the header
        f.write('PROCAR lm decomposed\n')
        # # of k-points:  165         # of bands:    6         # of ions:    10
        f.write('# of k-points:  ' + str(self.Nkpoints) + '         # of bands:    '
                + str(self.Nbands) + '         # of ions:    ' + str(self.Natoms) + '\n')
    
        for ikpoint in range(self.Nkpoints):
            # preparing the K-point line
            index = str(ikpoint+1)
            kpoint = kpoints[ikpoint][:3]
            kpoint = ' '.join([x for x in kpoint])
            weight = kpoints[ikpoint][3]
            kstr = '\n k-point ' + index + ' :   ' + kpoint +  ' weight = ' + weight + '\n'
            f.write(kstr)
            
      
            for iband in range(self.Nbands):
                # preparing the band line
                index = str(iband + 1)
                energy = str(bands[ikpoint][iband])
                occ = str(occupancies[ikpoint][iband])
                bstr = '\nband ' + index + ' # energy ' + energy + ' # occ. ' + occ + '\n'
                f.write(bstr)
      
                # preparing atom string
                # ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot
                astr = '\nion   ' + '  '.join(self.orbDict.keys()) + ' tot\n '
                for iatom in range(self.Natoms):
                    index = str(iatom + 1)
                    astr += index + ' ' + ' '.join(self.spd[ikpoint, iband, iatom])
                    astr+= ' ' + tot_orbs[ikpoint, iband, iatom] + '\n ' 
                # special line `tot`
                astr += 'tot ' + ' '.join(tot_atoms[ikpoint, iband])
                astr += ' ' + tot_oa[ikpoint, iband]  + '\n'
                f.write(astr)
        f.close()



class DFTB_bands:
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        self.filename =  filename
        self.Nkpoints = None
        self.Nbands = None
        self.Bands = None
        self.Occupancies = None
        
        return
  
    def load(self):
        data = open(self.filename, 'r')
        data = data.read()
    
        # Spin-polarized data is not supproted yet, if there exist, we
        # need to raise an exception
        spinful = re.findall(r'SPIN\s*2', data)
        if len(spinful) == 2:
            raise RuntimeError("Spinful calculations are not supported")
        
        # splitting by k-point
        #  KPT            1  SPIN            1  KWEIGHT    1.0000000000000000
        kpoints = re.split(r'\s*KPT\s*\d+\s*\w+\s*1\s*\w+\s*[-.0-9E]+\s*', data)
        # the first value has to be empty ('') there is nothing before the
        # first match
        if kpoints[0] != '':
            print("WARNING, the first match wasn\'t empty. It content is: ", kpoints[0])
        kpoints.pop(0)
        self.Nkpoints = len(kpoints)
        if self.verbose:
            print('Number of Kpoints (bands): ', self.Nkpoints)
        
        # splitting the numbers at each kpoint, only the numbers at the
        # middle and end are needed :
        #
        #     1   -20.981  2.00000
        evalues = []
        occs = [] # occupancies
        for kpoint in kpoints:
            occ = re.findall(r'\d+\s+[0-9.\-]+\s+([0-9.\-]+)\s*\n?', kpoint)
            kpoint = re.findall(r'\d+\s+([0-9.\-]+)\s+[0-9.\-]+\s*\n?', kpoint)
            #kpoint = np.array(kpoint, dtype=float)
            evalues.append(kpoint)
            occs.append(occ)
        evalues = np.array(evalues, dtype=float)
        occs = np.array(occs, dtype=float)
        if self.verbose:
            print('Bands found (Nkpoints. Nbands): ', evalues.shape)
    
        # # removed
        # evalues = evalues - self.eFermi
        # eFermi = 0
        self.Bands = evalues
        self.Occupancies = occs
        return

class DFTB_utils:
    """Utilities that do not belong to other place"""
    def __init__(self, verbose=True):
        self.verbose = verbose
        return
  
    def find_fermi(self, filename):
        # checking if the file exists
        import os
        if os.path.isfile(filename):
            fermiFile = open(filename, 'r').read()
        else:
            raise RuntimeError('File ' + filename + ' not found')
        # Fermi level:                        -0.1467617917 H           -3.9936 eV
        efermi = re.findall(r'Fermi level:\s*[\-\d.]+\s*H\s*([\-\d.]+)', fermiFile)
        efermi = float(efermi[0])
        if self.verbose:
            print("Fermi energy found: ", efermi, 'eV')
        return efermi
  
    def get_kpoints(self, filename):
        # in direct coordinates!
        
        # checking if the file exists
        from xml.dom import minidom
    
        dom = minidom.parse(filename)    
        kpoints = dom.getElementsByTagName('kpointsandweights')[0].firstChild.data
        kpoints = re.findall(r'-?\d+\.\d+', kpoints)
        kpoints = np.array(kpoints, dtype=float)
        Nkpoints = len(kpoints)/4
        if Nkpoints.is_integer:
            Nkpoints = int(Nkpoints)
        else:
            raise RuntimeError('non-integer number of kpoints')
        # the last enty is the weigth
        kpoints.shape = (Nkpoints, 4)
        if self.verbose:
            print('Nkpoints: ', Nkpoints)
        # print(kpoints)
        return kpoints

    def find_lattice(self, filename):
        # in direct coordinates!
        # checking if the file exists
        from xml.dom import minidom
    
        dom = minidom.parse(filename)    
        lat = dom.getElementsByTagName('latticevectors')[0].firstChild.data
        lat = re.findall(r'-?\d+\.\d+', lat)
        lat = np.array(lat, dtype=float)
        lat.shape = (3,3)
        # Bohr to Angstroms
        lat = lat*0.529177249 
        return lat

  
    def writeOutcar(self):
        f = open('OUTCAR', 'w')
        efermi = self.find_fermi(args.detailed)
        f.write('E-fermi : ' + str(efermi) + ' \n')
        
        lat = self.find_lattice(args.kpointsfile)
        # print('lat', lat)
        vol = np.dot( lat[0], np.cross(lat[1], lat[2]) )
        b0 = np.cross(lat[1], lat[2])/vol
        b1 = np.cross(lat[2], lat[0])/vol
        b2 = np.cross(lat[0], lat[1])/vol
        
        f.write('\nreciprocal lattice vectors \n')
        # print('foo')
        f.write(str(lat[0,0]) + ' ' + str(lat[0,1]) + ' ' +  str(lat[0,2]) + '  ' )
        f.write(str(b0[0])    + ' ' + str(b0[1])    + ' ' +  str(b0[2])    + '\n')
        f.write(str(lat[1,0]) + ' ' + str(lat[1,1]) + ' ' +  str(lat[1,2]) + '  ' )
        f.write(str(b1[0])    + ' ' + str(b1[1])    + ' ' +  str(b1[2])    + '\n')
        f.write(str(lat[2,0]) + ' ' + str(lat[2,1]) + ' ' +  str(lat[2,2]) + '  ' )
        f.write(str(b2[0])    + ' ' + str(b2[1])    + ' ' +  str(b2[2])    + '\n')
        f.close()
      
    
class DFTBParser:
    """The class parses the input form DFTB+.
    
    Parameters
    ----------

    dirname : str, optional
        Directory path to where calculation took place, by default ""
    eigenvec_filename : str, optional
        The (plain-text) eigenvectors, by default 'eigenvec.out'
    bands_filename : str, optional
        The file with the bands, by default 'band.out'
    detailed_out : str, optional 
        The file with the Fermi energy, by default 'detailed.out'
    detailed_xml : str, optional
        The file with the list of kpoints, by default 'detailed.xml'

    

    """
    def __init__(self,
                 dirname:str = '',
                 eigenvec_filename:str = 'eigenvec.out',
                 bands_filename:str = 'band.out',
                 detailed_out:str = 'detailed.out',
                 detailed_xml:str = 'detailed.xml'
                 ):
        
        self.evecs = DFTB_evec(filename = eigenvec_filename,
                               verbose = False)
        self.evecs.load()

        # I need the bands
        bands = DFTB_bands(filename=bands_filename, verbose=False)
        bands.load()
        self.evecs.set_bands(bands=bands.Bands, occupancies=bands.Occupancies)
        
        # I need the kpoints
        utils = DFTB_utils(verbose=False)
        kpoints = utils.get_kpoints(filename = detailed_xml)
        self.evecs.set_kpoints(kpoints = kpoints)

        # I need the Fermi level
        efermi = utils.find_fermi(filename = detailed_out)

        # and the lattice
        lat = utils.find_lattice(filename = detailed_xml)
        
        # 
        # No standard format for the DOS, it is up to the user. I won't parse it
        #       
        self.dos = None

        reciprocal_lattice = np.zeros_like(lat)
        a = lat[0, :]
        b = lat[1, :]
        c = lat[2, :]
        volume = np.dot(a,np.cross(b, c))
        a_star = (2 * np.pi) * np.cross(b, c) / volume
        b_star = (2 * np.pi) * np.cross(c, a) / volume
        c_star = (2 * np.pi) * np.cross(a, b) / volume
        reciprocal_lattice[0, :] = a_star
        reciprocal_lattice[1, :] = b_star
        reciprocal_lattice[2, :] = c_star



        spd = self.evecs.spd
        shape = spd.shape
        spd.shape = (shape[0], # kpoints
                     shape[1], # bands
                     shape[2], # atoms
                     1,        # unknown  
                     shape[3], # orbitals
                     1)        # spin
        phase = self.evecs.phase
        phase.shape = (shape[0],
                       shape[1],
                       shape[2],
                       1,
                       shape[3],
                       1)
        print('spd shape, ', self.evecs.spd.shape)
        print('phase shape, ', self.evecs.phase.shape)
        
        self.ebs = ElectronicBandStructure(
            kpoints=self.evecs.kpoints,
            bands=self.evecs.bands,
            projected=spd,
            efermi=efermi,
            kpath=None,
            projected_phase=phase,
            reciprocal_lattice=reciprocal_lattice,
        )

        self.structure = None
        self.kpath = None
        
        return
