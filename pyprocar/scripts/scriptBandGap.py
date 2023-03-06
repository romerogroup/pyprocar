import numpy as np

from ..io import AbinitParser
from ..io import ElkParser
from ..io import ProcarParser
from ..io import LobsterParser
from ..utils import UtilsProcar

from .. import io


def getFermi(procar:str, code:str, outcar:str):
    """A function to get the fermi energy

    Parameters
    ----------
    procar : str
        The PROCAR filename
    code : str
        The code name
    outcar : str
        The OUTCAR filename

    Returns
    -------
    float
        The fermi energy in ev
    """
    
    fermi = None

    if code == "vasp":
        # Parses through Bands in PROCAR
        procarFile = ProcarParser()
        procarFile.readFile(procar=procar)
        if outcar:
            outcarparser = UtilsProcar()
            if fermi is None:
                fermi = outcarparser.FermiOutcar(outcar)
                print("Fermi energy found in OUTCAR file = %s eV" % str(fermi))
        else:
            print("ERROR: OUTCAR Not Found")

    elif code == "elk":
        if fermi is None:
            procarFile = ElkParser()
            procarFile.readFile()
            fermi = procarFile.fermi
            print("Fermi energy found in Elk output file = %s eV " % str(fermi))

    elif code == "abinit":
        procarFile = ProcarParser()
        procarFile.readFile(procar=procar)
        if fermi is None:
            abinitFile = AbinitParser(abinit_output=outcar)
            fermi = abinitFile.fermi
            print("Fermi energy found in Abinit output file = %s eV" % str(fermi))

    elif code == "qe":
        if fermi is None:
            procarFile = QEParser()
            fermi = procarFile.fermi
            print("Fermi energy   :  %s eV (from Quantum Espresso output)" % str(fermi))

    elif code == "lobster":
        if fermi is None:
            fermi = procarFile.fermi
            print("Fermi energy   :  %s eV (from Lobster output)" % str(fermi))
            # lobster already shifts fermi so we set it to zero here.
            fermi = 0.0

    return fermi


def bandgap(procar:str=None, outcar:str=None, code:str="vasp", fermi:float=None, repair:bool=True):
    """A function to find the band gap

    Parameters
    ----------
    procar : str, optional
        The PROCAR filename, by default None
    outcar : str, optional
        The OUTCAR filename, by default None
    code : str, optional
        The code name, by default "vasp"
    fermi : float, optional
        The fermi energy, by default None
    repair : bool, optional
        Boolean to repair the PROCAR file, by default True

    Returns
    -------
    float
        Returns the bandgap energy
    """

    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(procar, procar)
            print("PROCAR repaired. Run with repair=False next time.")

    bandGap = None

    if fermi is None:
        fermi = getFermi(procar, code, outcar)

    if code == "vasp":
        procarFile = ProcarParser()
        procarFile.readFile(procar=procar)

    elif code == "abinit":
        procarFile = ProcarParser()
        procarFile.readFile(procar=procar)

    elif code == "elk":
        procarFile = ElkParser()

    elif code == "qe":
        procarFile = QEParser()

    elif code == "lobsters":
        procarFile = LobsterParser()

    bands = np.array(procarFile.bands)
    subBands = np.subtract(bands, fermi)

    negArr = subBands[subBands < 0]
    posArr = subBands[subBands > 0]

    negVal = np.amax(negArr)
    posVal = np.amin(posArr)

    idx = np.where(subBands == negVal)[1][0]

    if all(i >= 0 for i in subBands[:, idx]) or all(i <= 0 for i in subBands[:, idx]):
        possibleGap = posVal - negVal
        if bandGap is None:
            bandGap = possibleGap
        elif possibleGap < bandGap:
            bandGap = possibleGap
    else:
        bandGap = 0

    print("Band Gap = %s eV " % str(bandGap))

    return bandGap
