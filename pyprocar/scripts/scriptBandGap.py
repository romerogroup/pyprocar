import numpy as np

from pyprocar.io import Parser


def bandgap(
    _procar: str | None = None,
    dirname: str | None = None,
    _outcar: str | None = None,
    code: str = "vasp",
    fermi: float | None = None,
    _repair: bool = True,
) -> float | None:
    """A function to find the band gap

    Parameters
    ----------
    _procar : str, optional
        The PROCAR filename, by default None
    dirname : str, optional
        The directory path, by default None
    _outcar : str, optional
        The OUTCAR filename, by default None
    code : str, optional
        The code name, by default "vasp"
    fermi : float, optional
        The fermi energy, by default None
    _repair : bool, optional
        Boolean to repair the PROCAR file, by default True

    Returns
    -------
    float
        Returns the bandgap energy
    """
    bandGap: float | None = None

    parser = Parser(code=code, dirpath=dirname if dirname is not None else ".")
    ebs = parser.ebs
    if ebs is None:
        return None

    if fermi is None:
        fermi = ebs.fermi

    bands = np.array(ebs.bands)
    subBands = np.subtract(bands, fermi)

    negArr = subBands[subBands < 0]
    posArr = subBands[subBands > 0]

    negVal = np.amax(negArr)
    posVal = np.amin(posArr)

    idx = np.where(subBands == negVal)[1][0]

    if all(i >= 0 for i in subBands[:, idx]) or all(i <= 0 for i in subBands[:, idx]):
        possibleGap = posVal - negVal
        if bandGap is None or possibleGap < bandGap:
            bandGap = possibleGap
    else:
        bandGap = 0

    print("Band Gap = %s eV " % str(bandGap))

    return bandGap
