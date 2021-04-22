from re import findall
from numpy import array


class AbinitParser:
    """
    This class contains methods to parse the fermi energy and reciprocal
    lattice vectors from the Abinit output file.
    """

    def __init__(self, abinit_output=None):

        self.abinit_output = abinit_output
        self.fermi = None
        self.reclat = None  # reciprocal lattice vectors
        self.nspin = None  # spin

        self._readFermi()
        self._readRecLattice()

        return

    def _readFermi(self):
        rf = open(self.abinit_output, "r")
        data = rf.read()
        self.fermi = float(
            findall("Fermi\w*.\(\w*.HOMO\)\s*\w*\s*\(\w*\)\s*\=\s*([0-9.+-]*)", data)[0]
        )

        # Converting from Hartree to eV
        self.fermi = 27.211396641308 * self.fermi

        # read spin (nsppol)
        self.nspin = findall(r"nsppol\s*=\s*([1-9]*)", data)[0]

        rf.close()

    def _readRecLattice(self):
        rf = open(self.abinit_output, "r")
        data = rf.read()
        rf.close()
        lattice_block = findall(r"G\([1,2,3]\)=\s*([0-9.\s-]*)", data)
        if len(lattice_block) > 3:
            lattice_block = lattice_block[3:]
        self.reclat = array(
            [lattice_block[0:3][i].split() for i in range(len(lattice_block))],
            dtype=float,
        )
