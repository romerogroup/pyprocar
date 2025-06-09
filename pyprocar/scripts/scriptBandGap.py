import numpy as np

from .. import io


def bandgap(
    procar: str = None,
    dirname: str = None,
    outcar: str = None,
    code: str = "vasp",
    fermi: float = None,
    repair: bool = True,
):
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

    bandGap = None

    parser = io.Parser(code=code, dirpath=dirname)
    ebs = parser.ebs

    if fermi is None:
        fermi = ebs.efermi

    bands = np.array(ebs.bands)
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
