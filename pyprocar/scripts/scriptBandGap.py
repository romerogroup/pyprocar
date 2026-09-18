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
    # Bands are expected with shape (nkpoints, nbands, nspins). The gap is
    # evaluated over all spin channels: VBM is the highest occupied state and
    # CBM the lowest unoccupied state among all spins.
    if subBands.ndim == 2:
        subBands = subBands[..., np.newaxis]

    negArr = subBands[subBands < 0]
    posArr = subBands[subBands > 0]

    negVal = np.amax(negArr)
    posVal = np.amin(posArr)

    # The system is metallic if, in any spin channel, the band holding the
    # highest occupied state of that channel crosses the Fermi level.
    is_metal = False
    for ispin in range(subBands.shape[-1]):
        spin_bands = subBands[:, :, ispin]
        spin_neg = spin_bands[spin_bands < 0]
        if spin_neg.size == 0:
            continue
        idx = np.where(spin_bands == np.amax(spin_neg))[1][0]
        if not (np.all(spin_bands[:, idx] >= 0) or np.all(spin_bands[:, idx] <= 0)):
            is_metal = True
            break

    if not is_metal:
        possibleGap = posVal - negVal
        if bandGap is None:
            bandGap = possibleGap
        elif possibleGap < bandGap:
            bandGap = possibleGap
    else:
        bandGap = 0

    bandGap = float(bandGap)

    print("Band Gap = %s eV " % str(bandGap))

    return bandGap
