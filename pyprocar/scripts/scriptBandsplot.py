_author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

from ..utils.info import orbital_names
from .. import io
from ..plotter import EBSPlot
from ..utils import welcome, ROOT


with open(os.path.join(ROOT,'pyprocar','cfg','band_structure.yml'), 'r') as file:
    plot_opt = yaml.safe_load(file)
    
def bandsplot(
    code: str,
    dirname: str,
    mode:str="plain",
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    items:dict={},
    fermi:float=None,
    fermi_shift:float=0,
    interpolation_factor:int=1,
    interpolation_type:str="cubic",
    projection_mask:np.ndarray=None,
    kticks=None,
    knames=None,
    kdirect:bool=True,
    elimit: List[float]=None,
    ax:plt.Axes=None,
    show:bool=True,
    savefig:str=None,
    print_plot_opts:bool=False,
    **kwargs
    ):
    """A function to plot the band structutre

    Parameters
    ----------
    code : str, optional
        String to of the code used, by default "vasp"
    dirname : str, optional
        The directory name of the calculation, by default None
    mode : str, optional
        Sting for the mode of the calculation, by default "plain"
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals, by default {}
    fermi : float, optional
        Float for the fermi energy, by default None. By default the fermi energy will be shifted by the fermi value that is found in the directory. 
        For band structure calculations, due to convergence issues, this fermi energy might not be accurate. If so add the fermi energy from the self-consistent calculation. 
    fermi_shift : float, optional
        Float to shift the fermi energy, by default 0.
    interpolation_factor : int, optional
        The interpolation_factor, by default 1
    interpolation_type : str, optional
        The interpolation type, by default "cubic"
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    kticks : _type_, optional
        A list of kticks, by default None
    knames : _type_, optional
        A list of kanems, by default None
    elimit : List[float], optional
        A list of floats to decide the energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes, by default None
    show : bool, optional
        Boolean if to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    print_plot_opts: bool, optional
        Boolean to print the plotting options
    """

    modes=["plain","parametric","scatter","atomic",
           "overlay", "overlay_species", "overlay_orbitals", "ipr"]
    modes_txt=' , '.join(modes)
    message=f"""
            --------------------------------------------------------
            There are additional plot options that are defined in a configuration file. 
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}
            --------------------------------------------------------
            """
    print(message)
    if print_plot_opts:
        for key,value in plot_opt.items():
            print(key,':',value)

    parser = io.Parser(code = code, dir = dirname)
    ebs = parser.ebs
    structure = parser.structure
    kpath = parser.kpath

    # shifting fermi to 0
    if fermi is None:
        fermi=ebs.efermi
        print("""
            WARNING : Fermi Energy not set! Set `fermi={value}`. By default, shifting by fermi energy found in current directory.
            --------------------------------------------------------
            """
        )
    ebs.bands -= fermi

    ebs.bands += fermi_shift
    fermi_level = fermi_shift


    # fixing the spin, to plot two channels into one (down is negative)
    if np.array_equal(spins, [-1,1]) or np.array_equal(spins, [1,-1]):
        if ebs.fix_collinear_spin():
          spins = [0]
        
    ebs_plot = EBSPlot(ebs, kpath, ax, spins, kdirect=kdirect ,**kwargs)

 
    labels = []
    if mode == "plain":
        ebs_plot.plot_bands()

    elif mode == "ipr":
      weights = ebs_plot.ebs.ebs_ipr()
      if plot_opt['weighted_color']['value']:
        color_weights = weights
      else:
        color_weights = None
      if plot_opt['weighted_width']['value']:
        width_weights = weights
      else:
        width_weights = None
      color_mask = projection_mask
      width_mask = projection_mask

      ebs_plot.plot_parameteric(
        color_weights=color_weights,
        width_weights=width_weights,
        color_mask=color_mask,
        width_mask=width_mask,
        spins=spins,
        elimit=elimit,  
        )
      ebs_plot.set_colorbar_title(title='Inverse Participation Ratio')
      
        
    elif mode in ["overlay", "overlay_species", "overlay_orbitals"]:
        weights = []
        
        if mode == "overlay_species":
            for ispc in structure.species:
                labels.append(ispc)
                atoms = np.where(structure.atoms == ispc)[0]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)
        if mode == "overlay_orbitals":
            for iorb,orb in enumerate(["s", "p", "d", "f"]):
                if orb == "f" and not ebs_plot.ebs.norbitals > 9:
                    continue
                orbitals = orbital_names[orb]
                labels.append(orb)
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)

        elif mode == "overlay":
            if isinstance(items, dict):
                items = [items]

            if isinstance(items, list):
                for it in items:
                    for ispc in it:
                        atoms = np.where(structure.atoms == ispc)[0]
                        if isinstance(it[ispc][0], str):
                            orbitals = []
                            for iorb in it[ispc]:
                                orbitals = np.append(orbitals, orbital_names[iorb]).astype(int)
                            labels.append(ispc + "-" + "".join(it[ispc]))
                        else:
                            orbitals = it[ispc]
                            labels.append(ispc + "-" + "_".join(str(x) for x in it[ispc]))
                        w = ebs_plot.ebs.ebs_sum(
                            atoms=atoms,
                            principal_q_numbers=[-1],
                            orbitals=orbitals,
                            spins=spins,
                        )
                        weights.append(w)
        ebs_plot.plot_parameteric_overlay(spins=spins,weights=weights)
    else:
        if atoms is not None and isinstance(atoms[0], str):
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(structure.atoms == iatom)[0]).astype(
                    np.int
                )

        if orbitals is not None and isinstance(orbitals[0], str):
            orbital_str = orbitals

            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(orbitals, orbital_names[iorb]).astype(np.int)


        weights = ebs_plot.ebs.ebs_sum(atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins)
        if plot_opt['weighted_color']['value']:
            color_weights = weights
        else:
            color_weights = None
        if plot_opt['weighted_width']['value']:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask
        if mode == "parametric":
            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins
                )
            ebs_plot.set_colorbar_title()
        elif mode == "scatter":
            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins
            )
            ebs_plot.set_colorbar_title()
        elif mode == "atomic":
            if ebs.kpoints.shape[0]!=1:
                raise Value('Must use a single kpoint')
            if color_weights is not None:
                color_weights = np.vstack((color_weights, color_weights))
            ebs_plot.plot_atomic_levels(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                elimit=elimit)
            
            ebs_plot.set_xlabel(label='')
            ebs_plot.set_colorbar_title()

        else:
            print("Selected mode %s not valid. Please check the spelling " % mode)
            
    ebs_plot.set_xticks(kticks, knames)
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    ebs_plot.draw_fermi(fermi_level=fermi_level)
    ebs_plot.set_ylabel()


    ebs_plot.set_title()
    ebs_plot.grid()
    ebs_plot.legend(labels)
    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()
        
    return ebs_plot.fig, ebs_plot.ax
