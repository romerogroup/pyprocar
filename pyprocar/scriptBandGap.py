import numpy as np
from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .procarparser import ProcarParser
from .qeparser import QEParser
from .lobsterparser import LobsterParser
from .utilsprocar import UtilsProcar


def getFermi(procar, code, outcar):  # from ScriptsBandPlot made into method
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


def bandgap(procar=None, outcar=None, code="vasp", fermi=None, repair=True):

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
