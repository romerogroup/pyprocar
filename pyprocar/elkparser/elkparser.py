"""
Created on Fri March 2 2020
@author: Pedram Tavadze

"""

from re import findall
from numpy import array, linspace, where, zeros, dot



class ElkParser():
    def __init__(self, elkin = 'elk.in', kdirect = True):

    
        # elk specific inp
        self.fin = elkin
        self.file_names = []
        self.high_symmetry_points = None
        self.nhigh_sym = None
        self.nspecies = None
        self.nbands = None
        self.composition = {}
        self.tasks = None
        self.kticks = None
        self.knames = None
        self.kdirect = kdirect


        self.kpoints = None
        self.kpointsCount = None
        self.bands = None
        self.bandsCount = None
        self.ionsCount = None

        
        self.spd = None
        self.cspd = None
        
        self.fermi = None
        self.reclat = None #reciprocal lattice vectors
        
        self.orbitalName = [
            "Y00", "Y1-1", "Y10", "Y11", "Y2-2", "Y2-1", "Y20", "Y21", "Y22", "Y3-3"
            ,"Y3-2","Y3-1","Y30","Y3-1","Y3-2","Y3-3"]
        self.orbitalName_short = ["s","p","d","f"]
        self.orbitalCount = None  #number of orbitals

        # number of spin components (blocks of data), 1: non-magnetic non
        # polarized, 2: spin polarized collinear, 4: non-collinear
        # spin.
        # NOTE: before calling to `self._readOrbital` the case '4'
        # is marked as '1'
        self.ispin = None
        
        rf = open(self.fin,'r')
        self.elkIn = rf.read()
        rf.close()

        self._readElkin()
        self._readFiles()
        
        self._readFermi()
        self._readRecLattice()

        return

    @property
    def nspin(self):
        """
        number of spin, default is 1.
        """
        nspindict = {1: 1, 2: 2, 4: 2, None: 1}
        return nspindict[self.ispin]


    @property
    def spd_orb(self):
        # indices: ikpt, iband, ispin, iion, iorb
        # remove indices and total from iorb.
        return self.spd[:, :, :, 1:-1]


    def _readElkin(self):
        self.nhigh_sym,self.kpointsCount = [int(x) for x in findall('plot1d\n\s*([0-9]*)\s*([0-9]*)',self.elkIn)[0]]
        raw_hsp = findall('plot1d\n\s*[0-9]*\s*[0-9]*.*\n'+self.nhigh_sym*'([0-9\s\.-]*).*\n',self.elkIn)[0]
        self.high_symmetry_points = zeros(shape=(self.nhigh_sym,3))
        for ihs in range(self.nhigh_sym):
            self.high_symmetry_points[ihs,:] = [float(x) for x in raw_hsp[ihs].split()[0:3]]             
        self.nspecies = int(findall('atoms\n\s*([0-9]*)',self.elkIn)[0])
        self.ionsCount = 0
        for ispc in findall('\'([A-Za-z]*).in\'.*\n\s*([0-9]*)',self.elkIn):
            self.composition[ispc[0]] =  int(ispc[1])
            self.ionsCount+=int(ispc[1])
        raw_ticks = findall('plot1d\n\s*[0-9]*\s*[0-9]*.*\n'+self.nhigh_sym*'.*:(.*)\n',self.elkIn)[0]
        if len(raw_ticks) != self.nhigh_sym:
            self.knames = [str(x) for x in range(self.nhigh_sym)]
        else :
            self.knames = ["$%s$" % (x.replace(',','').replace('vlvp1d','').replace(' ','')) for x in findall('plot1d\n\s*[0-9]*\s*[0-9]*.*\n'+self.nhigh_sym*'.*:(.*)\n',self.elkIn)[0]]
        self.tasks = [int(x) for x in findall('tasks\n\s*([0-9\s\n]*)',self.elkIn)[0].split()]

        if 20 in self.tasks :
            self.file_names.append('BANDS.OUT')
        if 21 in self.tasks or 22 in self.tasks :
            ispc = 1
            for spc in self.composition:
                for iatom in range(self.composition[spc]):
                    self.file_names.append('BAND_S{:02d}_A{:04d}.OUT'.format(ispc,iatom+1))
                ispc += 1
    
    def _readFiles(self):
        rf = open(self.file_names[0],'r')
        lines = rf.readlines()
        rf.close()
        self.bandsCount = int(len(lines)/(self.kpointsCount+1))
        self.bands = zeros(shape=(self.kpointsCount,self.bandsCount))
        rf = open('BANDLINES.OUT','r')
        bandLines = rf.readlines()
        rf.close()
        tick_pos = []
        # using strings for a better comparision and avoiding rounding by python
        for iline in range(0,len(bandLines),3):
            tick_pos.append(bandLines[iline].split()[0])
        x_points = []
        for iline in range(self.kpointsCount):
            x_points.append(lines[iline].split()[0])
        
        self.kpoints = zeros(shape=(self.kpointsCount,3))
        x_points = array(x_points)
        tick_pos = array(tick_pos)
        

        self.kticks = []
        for ihs in range(1,self.nhigh_sym):
            # if ihs == 1:
            #     start = 0
            # else:
            
            start = where(x_points == tick_pos[ihs-1])[0][0]
            end = where(x_points == tick_pos[ihs])[0][0]+1
            self.kpoints[start:end][:] = linspace(self.high_symmetry_points[ihs-1],self.high_symmetry_points[ihs],end-start)
            self.kticks.append(start)
        self.kticks.append(self.kpointsCount-1) 
        
        rf = open(self.file_names[0],'r')
        lines = rf.readlines()
        rf.close()

        iline = 0
        for iband in range(self.bandsCount):
            for ikpoint in range(self.kpointsCount):
                self.bands[ikpoint,iband] =  float(lines[iline].split()[1])
                iline += 1
            if ikpoint == self.kpointsCount -1 :
                iline +=1
        self.bands *= 27.21138386
        self.spinCount = 1
        self.orbitalCount = 16
        self.spd = zeros(shape=(self.kpointsCount,self.bandsCount,self.spinCount,self.ionsCount+1,self.orbitalCount+2))
        idx_bands_out = None
        for ifile in range(len(self.file_names)):
            if self.file_names[ifile] == 'BANDS.OUT':
                idx_bands_out = ifile
        if idx_bands_out!= None:
            del self.file_names[idx_bands_out]


        for ifile in range(self.ionsCount):
            rf = open(self.file_names[ifile],'r')
            lines = rf.readlines()
            rf.close()
            iline = 0

            for iband in range(self.bandsCount):
                for ikpoint in range(self.kpointsCount):
                    temp = array([float(x) for x in lines[iline].split()])
                    self.spd[ikpoint,iband,0,ifile,0] = ifile+1
                    self.spd[ikpoint,iband,0,ifile,1:-1] = temp[2:]
                    iline += 1
                if ikpoint == self.kpointsCount -1 :
                    iline +=1
        #self.spd[:,:,:,-1,:] = self.spd.sum(axis=3)
        self.spd[:,:,:,:,-1] = self.spd.sum(axis=4)
        self.spd[:,:,:,-1,:] = self.spd.sum(axis=3)
        self.spd[:,:,0,-1,0] = 0
    
    def _readFermi(self):
        rf = open('EFERMI.OUT','r')
        self.fermi = float(rf.readline().split()[0])
        rf.close()
        
    def _readRecLattice(self):
        rf = open('LATTICE.OUT','r')
        data = rf.read()
        rf.close()
        lattice_block = findall(r'matrix\s*:([\s0-9.]*)Inverse',data)
        lattice_array = array(lattice_block[1].split(),dtype=float)        
        self.reclat = lattice_array.reshape((3,3))
        
        if self.kdirect is False:
            self.kpoints = dot(self.kpoints, self.reclat)