__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt

from .. import io
from ..plotter import EBSPlot
from ..utils import welcome
from ..utils.defaults import settings

def bandsplot_2d(
    code="vasp",
    dirname:str=None,
    lobster:bool=False,
    mode:str="plain",
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    items:dict={},
    fermi:float=None,
    interpolation_factor:int=1,
    interpolation_type:str="cubic",
    projection_mask=None,
    vmax:float=None,
    vmin:float=None,
    kticks=None,
    knames=None,
    kdirect:bool=True,
    elimit: List[float]=None,
    ax:plt.Axes=None,
    title:str=None,
    show:bool=True,
    savefig:str=None,
    **kwargs,
    ):
    """A function to plot the 2d bandstructure

    Parameters
    ----------
    code : str, optional
        The code name, by default "vasp"
    dirname : str, optional
        The directory name where the calculation is, by default None
    lobster : bool, optional
        Boolean if this is a lobster calculation, by default False
    mode : str, optional
        String for mode type, by default "plain"
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals , by default {}
    fermi : float, optional
        Ther fermi energy, by default None
    interpolation_factor : int, optional
        The interpolation factor, by default 1
    interpolation_type : str, optional
        The interpolation type, by default "cubic"
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    vmax : float, optional
        Value to normalize the minimum projection value., by default None, by default None, by default None
    vmin : float, optional
        Value to normalize the maximum projection value., by default None, by default None, by default None
    kticks : _type_, optional
        The kitcks, by default None
    knames : _type_, optional
        The knames, by default None
    kdirect : bool, optional
        _description_, by default True
    elimit : List[float], optional
        The energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes objext, by default None
    title : str, optional
        String for the title name, by default None
    show : bool, optional
        Boolean to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    """


    # Turn interactive plotting off
    # plt.ioff()

    # Verbose section

    settings.modify(kwargs)

    parser = io.Parser(code = code, dir = dirname)
    ebs = parser.ebs
    structure = parser.structure
    kpath = parser.kpath
    
    return None

