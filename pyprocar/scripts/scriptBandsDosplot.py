__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import matplotlib.pyplot as plt

from pyprocar.cfg.base import PlotType
from pyprocar.cfg.dos import DensityOfStatesConfig
from pyprocar.scripts.scriptBandsplot import bandsplot
from pyprocar.scripts.scriptDosplot import dosplot
from pyprocar.utils import welcome


def bandsdosplot(
    bands_settings: dict | None = None,
    dos_settings: dict | None = None,
    dos_limit: list[int] = None,
    elimit: list[int] = None,
    k_limit=None,
    grid: bool = False,
    code: str = "vasp",
    savefig: str = None,
    title: str = None,
    title_fontsize: float = 16,
    draw_fermi: bool = True,
    show: bool = True,
    dpi: int = 300,
    figsize=(8, 6),
):
    """A function to plot the band structure and the density of states in the same plot

    Parameters
    ----------
    bands_settings : dict, optional
        A dictionary containing the keyword arguments from bandsplot, by default None
    dos_settings : dict, optional
         A dictionary containing the keyword arguments from dosplot, by default None
    dos_limit : List[int], optional
        The dos window to plot, by default None
    elimit : List[int], optional
        The energy window to plot, by default None
    k_limit : _type_, optional
        The kpath points to plot, by default None
    grid : bool, optional
        Boolean to plot a grid, by default False
    code : str, optional
        The code to use, by default "vasp"
    savefig : str, optional
        The filename to to save the plot as., by default None
    title : str, optional
        String for the title name, by default None
    title_fontsize : float, optional
        Float for the title size, by default 16
    draw_fermi : bool, optional
        Boolean to plot the fermi level, by default True
    show : bool, optional
        Boolean to show the plot, by default True
    """

    welcome()

    bands_settings = {**(bands_settings or {}), "code": code, "show": False}
    dos_settings = {
        **(dos_settings or {}),
        "code": code,
        "orientation": "vertical",
        "show": False,
    }

    fig, (ax_ebs, ax_dos) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)
    bandsplot(ax=ax_ebs, **bands_settings)
    dosplot(ax=ax_dos, **dos_settings)
    ax_dos.set_ylabel("")

    if elimit is not None:
        ax_ebs.set_ylim(elimit)
    if dos_limit is not None:
        ax_dos.set_xlim(dos_limit)
    if k_limit is not None:
        ax_ebs.set_xlim(k_limit)
    if grid:
        ax_ebs.grid()
        ax_dos.grid()
    if draw_fermi:
        fermi_style = DensityOfStatesConfig(plot_type=PlotType.DENSITY_OF_STATES)
        for ax in (ax_ebs, ax_dos):
            ax.axhline(
                y=0,
                color=fermi_style.fermi_color,
                linestyle=fermi_style.fermi_linestyle,
                linewidth=fermi_style.fermi_linewidth,
            )

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    if show:
        plt.show()

    return fig, ax_ebs, ax_dos
