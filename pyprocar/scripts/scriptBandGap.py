import numpy as np

from pyprocar.io import get_parser


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

    parser = get_parser(code, dirname)
    ebs = parser.ebs
    if ebs is None or ebs.bands is None:
        raise ValueError(f"No band structure found in {dirname}")

    if fermi is None:
        fermi = ebs.fermi

    bands = ebs.bands.value - fermi
    if bands.ndim == 2:
        bands = bands[..., np.newaxis]

    bandGap = 0.0 if _is_metal(bands) else float(bands[bands > 0].min() - bands[bands < 0].max())

    print("Band Gap = %s eV " % str(bandGap))

    return bandGap


def _is_metal(bands: np.ndarray) -> bool:
    for channel in np.moveaxis(bands, -1, 0):
        occupied = channel[channel < 0]
        if occupied.size == 0:
            continue
        highest_occupied_band = channel[:, np.where(channel == occupied.max())[1][0]]
        if (highest_occupied_band > 0).any() and (highest_occupied_band < 0).any():
            return True
    return False
