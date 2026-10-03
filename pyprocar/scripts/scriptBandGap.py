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
    band_crosses_fermi = (bands < 0).any(axis=0) & (bands > 0).any(axis=0)

    if band_crosses_fermi.any():
        bandGap = 0.0
    else:
        bandGap = float(bands[bands > 0].min() - bands[bands < 0].max())

    print("Band Gap = %s eV " % str(bandGap))

    return bandGap
