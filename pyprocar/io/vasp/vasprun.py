import collections
import logging
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import Structure

logger = logging.getLogger(__name__)


class VaspXML(collections.abc.Mapping):
    """A class to parse the vasprun xml file

    Parameters
    ----------
    filename : str, optional
        The vasprun.xml filename, by default "vasprun.xml"

    Raises
    ------
    ValueError
        File not found
    """

    non_colinear_spins_dict = {
        "spin 1": "Spin-Total",
        "spin 2": "Spin-x",
        "spin 3": "Spin-y",
        "spin 4": "Spin-z",
    }
    colinear_spins_dict = {"spin 1": "Spin-up", "spin 2": "Spin-down"}

    def __init__(self, filepath: Union[str, Path] = "vasprun.xml"):

        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.data = self._parse_vasprun(self.filepath)

    @property
    def has_dos(self):
        return "dos" in self.data["general"]

    @property
    def spins_dict(self):

        spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
        if len(spins) == 4:
            return self.non_colinear_spins_dict
        else:
            return self.colinear_spins_dict

    @property
    def is_noncolinear(self):
        spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
        if len(spins) == 4:
            return True
        else:
            return False

    @property
    def is_spin_polarized(self):
        spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
        if len(spins) == 4:
            return True
        else:
            return False

    @property
    def bands(self):
        """Parses the electronic bands

        Returns
        -------
        np.ndarray
            The electronic bands
        """
        spins = list(self.data["general"]["eigenvalues"]["array"]["data"].keys())
        kpoints_list = list(
            self.data["general"]["eigenvalues"]["array"]["data"]["spin 1"].keys()
        )
        eigen_values = {}
        nbands = len(
            self.data["general"]["eigenvalues"]["array"]["data"][spins[0]][
                kpoints_list[0]
            ][kpoints_list[0]]
        )
        nkpoints = len(kpoints_list)
        for ispin in spins:
            eigen_values[ispin] = {}
            eigen_values[ispin]["eigen_values"] = np.zeros(shape=(nbands, nkpoints))
            eigen_values[ispin]["occupancies"] = np.zeros(shape=(nbands, nkpoints))
            for ikpoint, kpt in enumerate(kpoints_list):
                temp = np.array(
                    self.data["general"]["eigenvalues"]["array"]["data"][ispin][kpt][
                        kpt
                    ]
                )
                eigen_values[ispin]["eigen_values"][:, ikpoint] = (
                    temp[:, 0] - self.fermi
                )
                eigen_values[ispin]["occupancies"][:, ikpoint] = temp[:, 1]
        return eigen_values

    @property
    def bands_projected(self):
        """Parse the band projections

        Returns
        -------
        np.ndarray
            The band projections
        """
        # projected[iatom][ikpoint][iband][iprincipal][iorbital][ispin]
        labels = self.data["general"]["projected"]["array"]["info"]
        spins = list(self.data["general"]["projected"]["array"]["data"].keys())
        kpoints_list = list(
            self.data["general"]["projected"]["array"]["data"][spins[0]].keys()
        )
        bands_list = list(
            self.data["general"]["projected"]["array"]["data"][spins[0]][
                kpoints_list[0]
            ][kpoints_list[0]].keys()
        )
        bands_projected = {"labels": labels}

        nspins = len(spins)
        nkpoints = len(kpoints_list)
        nbands = len(bands_list)
        norbitals = len(labels)
        natoms = self.initial_structure.natoms
        bands_projected["projection"] = np.zeros(
            shape=(nspins, nkpoints, nbands, natoms, norbitals)
        )
        for ispin, spn in enumerate(spins):
            for ikpoint, kpt in enumerate(kpoints_list):
                for iband, bnd in enumerate(bands_list):
                    bands_projected["projection"][ispin, ikpoint, iband, :, :] = (
                        np.array(
                            self.data["general"]["projected"]["array"]["data"][spn][
                                kpt
                            ][kpt][bnd][bnd]
                        )
                    )
        # ispin, ikpoint, iband, iatom, iorbital
        bands_projected["projection"] = np.swapaxes(bands_projected["projection"], 0, 3)
        # iatom, ikpoint, iband, ispin, iorbital
        bands_projected["projection"] = np.swapaxes(bands_projected["projection"], 3, 4)
        # iatom, ikpoint, iband, iorbital, ispin
        bands_projected["projection"] = bands_projected["projection"].reshape(
            natoms, nkpoints, nbands, 1, norbitals, nspins
        )

        return bands_projected

    def _get_dos_total(self):
        """A helper method to get the total density of states

        Returns
        -------
        tuple
            Returns the dos_total info as a dict and the a list of labels
        """
        spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
        energies = np.array(
            self.data["general"]["dos"]["total"]["array"]["data"][spins[0]]
        )[:, 0]
        dos_total = {"energies": energies}

        for spin_name in spins:
            dos_total[self.spins_dict[spin_name]] = np.array(
                self.data["general"]["dos"]["total"]["array"]["data"][spin_name]
            )[:, 1]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self, atoms=[]):
        """A helper method to get the projected density of states

        Parameters
        ----------
        atoms : list, optional
            List of atoms, by default []

        Returns
        -------
        _type_
            Returns the dos_total info as a dict and the a list of labels
        """
        if len(atoms) == 0:
            atoms = np.arange(self.initial_structure.natoms)

        if "partial" in self.data["general"]["dos"]:
            dos_projected = {}
            ion_list = [
                "ion %s" % str(x + 1) for x in atoms
            ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                name = self.initial_structure.atoms[atoms[i]] + str(atoms[i])
                spins = list(
                    self.data["general"]["dos"]["partial"]["array"]["data"][
                        iatom
                    ].keys()
                )
                energies = np.array(
                    self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
                        spins[0]
                    ][spins[0]]
                )[:, 0]
                dos_projected[name] = {"energies": energies}
                for ispin in spins:
                    dos_projected[name][self.spins_dict[ispin]] = np.array(
                        self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
                            ispin
                        ][ispin]
                    )[:, 1:]
                    
   
                # if 'Spin-Total' in list(dos_projected[name].keys()):
                #     del dos_projected[name]['Spin-Total']
            return (
                dos_projected,
                self.data["general"]["dos"]["partial"]["array"]["info"],
            )
        else:
            print("This calculation does not include partial density of states")
            return None, None

    @cached_property
    def total_dos(self):
        total = []
        for ispin in self.dos_total:
            if ispin == "energies":
                continue
            total.append(self.dos_total[ispin])
        return np.array(total)

    @property
    def dos_to_dict(self):
        """
        The complete density (total,projected) of states as a python dictionary

        Returns
        -------
        dict
             The complete density (total,projected) of states as a python dictionary
        """

        return {"total": self._get_dos_total(), "projected": self._get_dos_projected()}

    @property
    def dos_total(self):
        """Returns the total dos dict

        Returns
        -------
        dict
            Returns the total dos dict
        """
        dos_total, labels = self._get_dos_total()
        return dos_total

    @property
    def dos_projected(self):
        """
        Returns the projected DOS as a multi-dimentional array, to be used in the
        pyprocar.core.dos object

        Returns
        -------
        np.ndarray
            Returns the projected DOS as a multi-dimentional array, to be used in the
            pyprocar.core.dos object
        """
        ret = []
        dos_projected, info = self._get_dos_projected()
        if dos_projected is None:
            return None
        norbitals = len(info) - 1
        info[0] = info[0].capitalize()
        labels = []
        labels.append(info[0])
        ret = []
        for iatom in dos_projected:
            temp_atom = []
            for iorbital in range(norbitals):
                temp_spin = []
                for key in dos_projected[iatom]:
                    if key == "energies":
                        continue
                    temp_spin.append(dos_projected[iatom][key][:, iorbital])
                temp_atom.append(temp_spin)
            ret.append(temp_atom)
        return np.array(ret)

    @property
    def kpoints(self):
        """
        Returns the kpoints used in the calculation in form of a pychemia.core.KPoints object

        Returns
        -------
        np.ndarray
            Returns the kpoints used in the calculation
            in form of a pychemia.core.KPoints object
        """

        if self.data["kpoints_info"]["mode"] == "listgenerated":
            kpoints = dict(
                mode="path", kvertices=self.data["kpoints_info"]["kpoint_vertices"]
            )
        else:
            kpoints = dict(
                mode=self.data["kpoints_info"]["mode"].lower(),
                grid=self.data["kpoints_info"]["kgrid"],
                shifts=self.data["kpoints_info"]["user_shift"],
            )
        return kpoints

    @property
    def kpoints_list(self):
        """
        Returns the dict of kpoints and weights used in the calculation in form of a pychemia.core.KPoints object

        Returns
        -------
        dict
            Returns a dict of kpoints information
            in form of a pychemia.core.KPoints object
        """
        return dict(
            mode="reduced",
            kpoints_list=self.data["kpoints"]["kpoints_list"],
            weights=self.data["kpoints"]["k_weights"],
        )

    @property
    def incar(self):
        """
        Returns the incar parameters used in the calculation as pychemia.code.vasp.VaspIncar object

        Returns
        -------
        Description
            Returns the incar parameters
        """
        return self.data["incar"]

    @property
    def vasp_parameters(self):
        """
        Returns all of the parameters vasp has used in this calculation
        """
        return self.data["vasp_params"]

    @property
    def potcar_info(self):
        """
        Returns the information about pseudopotentials(POTCAR) used in this calculation
        """
        return self.data["atom_info"]["atom_types"]

    @property
    def fermi(self):
        """
        Returns the fermi energy
        """
        
        if "fermi" in self.data["general"]["dos"]:
            return self.data["general"]["dos"]["fermi"]
        elif "efermi" in self.data["general"]["dos"]:
            return self.data["general"]["dos"]["efermi"]
        else:
            return None

    @property
    def species(self):
        """
        Returns the species in POSCAR
        """
        return self.initial_structure.species

    @property
    def structures(self):
        """
        Returns a list of pychemia.core.Structure representing all the ionic step structures
        """
        symbols = [x.strip() for x in self.data["atom_info"]["symbols"]]
        structures = []
        for ist in self.data["structures"]:

            st = Structure(
                atoms=symbols,
                fractional_coordinates=ist["reduced"],
                lattice=ist["cell"],
            )
            structures.append(st)
        return structures

    @property
    def structure(self):
        """
        crystal structure of the last step
        """
        return self.structures[-1]

    @property
    def forces(self):
        """
        Returns all the forces in ionic steps
        """
        return self.data["forces"]

    @property
    def initial_structure(self):
        """
        Returns the initial Structure as a pychemia structure
        """
        return self.structures[0]

    @property
    def final_structure(self):
        """
        Returns the final Structure as a pychemia structure
        """

        return self.structures[-1]

    @property
    def iteration_data(self):
        """
        Returns a list of information in each electronic and ionic step of calculation
        """
        return self.data["calculation"]

    @property
    def energies(self):
        """
        Returns a list of energies in each electronic and ionic step [ionic step,electronic step, energy]
        """
        scf_step = 0
        ion_step = 0
        double_counter = 1
        energies = []
        for calc in self.data["calculation"]:
            if "ewald" in calc["energy"]:
                if double_counter == 0:
                    double_counter += 1
                    scf_step += 1
                elif double_counter == 1:
                    double_counter = 0
                    ion_step += 1
                    scf_step = 1
            else:
                scf_step += 1
            energies.append([ion_step, scf_step, calc["energy"]["e_0_energy"]])
        return energies

    @property
    def last_energy(self):
        """
        Returns the last calculated energy of the system
        """
        return self.energies[-1][-1]

    @property
    def energy(self):
        """
        Returns the last calculated energy of the system
        """
        return self.last_energy

    @property
    def convergence_electronic(self):
        """
        Returns a boolian representing if the last electronic self-consistent
        calculation converged
        """
        ediff = self.vasp_parameters["electronic"]["EDIFF"]
        last_dE = abs(self.energies[-1][-1] - self.energies[-2][-1])
        if last_dE < ediff:
            return True
        else:
            return False

    @property
    def convergence_ionic(self):
        """
        Returns a boolian representing if the ionic part of the
        calculation converged
        """
        energies = np.array(self.energies)
        nsteps = len(np.unique(np.array(self.energies)[:, 0]))
        if nsteps == 1:
            print("This calculation does not have ionic steps")
            return True
        else:
            ediffg = self.vasp_parameters["ionic"]["EDIFFG"]
            if ediffg < 0:
                last_forces_abs = np.abs(np.array(self.forces[-1]))
                return not (np.any(last_forces_abs > abs(ediffg)))
            else:
                last_ionic_energy = energies[(energies[:, 0] == nsteps)][-1][-1]
                penultimate_ionic_energy = energies[(energies[:, 0] == (nsteps - 1))][
                    -1
                ][-1]
                last_dE = abs(last_ionic_energy - penultimate_ionic_energy)
                if last_dE < ediffg:
                    return True
        return False

    @property
    def convergence(self):
        """
        Returns a boolian representing if the the electronic self-consistent
        and ionic calculation converged
        """
        return self.convergence_electronic and self.convergence_ionic

    @property
    def is_finished(self):
        """
        Always returns True, need to fix this according to reading the xml as if the calc is
        not finished we will have errors in xml parser
        """
        # if vasprun.xml is read the calculation is finished
        return True

    def text_to_bool(self, text):
        """boolians in vaspxml are stores as T or F in str format, this function coverts them to python boolians"""
        text = text.strip(" ")
        if text == "T" or text == ".True." or text == ".TRUE.":
            return True
        else:
            return False

    def conv(self, ele, _type):
        """This function converts the xml text to the type specified in the attrib of xml tree"""

        if _type == "string":
            return ele.strip()
        elif _type == "int":
            return int(ele)
        elif _type == "logical":
            return self.text_to_bool(ele)
        elif _type == "float":
            if "*" in ele:
                return np.nan
            else:
                return float(ele)

    def get_varray(self, xml_tree):
        """Returns an array for each varray tag in vaspxml"""
        ret = []
        for ielement in xml_tree:
            tmp = []
            for x in ielement.text.split():
                try:
                    tmp.append(float(x))
                except ValueError:
                    tmp.append(0.0)
            ret.append(tmp)
        return ret

    def get_params(self, xml_tree, dest):
        """dest should be a dictionary
        This function is recurcive #check spelling"""
        for ielement in xml_tree:
            if ielement.tag == "separator":
                dest[ielement.attrib["name"].strip()] = {}
                dest[ielement.attrib["name"].strip()] = self.get_params(
                    ielement, dest[ielement.attrib["name"]]
                )
            else:
                if "type" in ielement.attrib:
                    _type = ielement.attrib["type"]
                else:
                    _type = "float"
                if ielement.text is None:
                    dest[ielement.attrib["name"].strip()] = None

                elif len(ielement.text.split()) > 1:
                    dest[ielement.attrib["name"].strip()] = [
                        self.conv(x, _type) for x in ielement.text.split()
                    ]
                else:
                    dest[ielement.attrib["name"].strip()] = self.conv(
                        ielement.text, _type
                    )

        return dest

    def get_structure(self, xml_tree):
        """Returns a dictionary of the structure"""
        ret = {}
        for ielement in xml_tree:
            if ielement.tag == "crystal":
                for isub in ielement:
                    if isub.attrib["name"] == "basis":
                        ret["cell"] = self.get_varray(isub)
                    elif isub.attrib["name"] == "volume":
                        ret["volume"] = float(isub.text)
                    elif isub.attrib["name"] == "rec_basis":
                        ret["rec_cell"] = self.get_varray(isub)
            elif ielement.tag == "varray":
                if ielement.attrib["name"] == "positions":
                    ret["reduced"] = self.get_varray(ielement)
        return ret

    def get_scstep(self, xml_tree):
        """This function extracts the self-consistent step information"""
        scstep = {"time": {}, "energy": {}}
        for isub in xml_tree:
            if isub.tag == "time":
                scstep["time"][isub.attrib["name"]] = [
                    float(x) for x in isub.text.split()
                ]
            elif isub.tag == "energy":
                for ienergy in isub:
                    scstep["energy"][ienergy.attrib["name"]] = float(ienergy.text)
        return scstep

    def get_set(self, xml_tree, ret):
        """This function will extract any element taged set recurcively"""
        if len(xml_tree) == 0:
            return ret
        if xml_tree[0].tag == "r":
            ret[xml_tree.attrib["comment"]] = self.get_varray(xml_tree)
            return ret
        else:
            ret[xml_tree.attrib["comment"]] = {}
            for ielement in xml_tree:

                if ielement.tag == "set":
                    ret[xml_tree.attrib["comment"]][ielement.attrib["comment"]] = {}
                    ret[xml_tree.attrib["comment"]][ielement.attrib["comment"]] = (
                        self.get_set(
                            ielement,
                            ret[xml_tree.attrib["comment"]][ielement.attrib["comment"]],
                        )
                    )
            return ret

    def get_general(self, xml_tree, ret):
        """This function will parse any element in calculatio other than the structures, scsteps"""
        if "dimension" in [x.tag for x in xml_tree]:
            ret["info"] = []
            ret["data"] = {}
            for ielement in xml_tree:
                if ielement.tag == "field":
                    ret["info"].append(ielement.text.strip(" "))
                elif ielement.tag == "set":
                    for iset in ielement:
                        ret["data"] = self.get_set(iset, ret["data"])
            return ret
        else:
            for ielement in xml_tree:
                if ielement.tag == "i":
                    if "name" in ielement.attrib:
                        if ielement.attrib["name"] == "fermi":
                            ret["fermi"] = float(ielement.text)
                    continue
                ret[ielement.tag] = {}
                ret[ielement.tag] = self.get_general(ielement, ret[ielement.tag])
            return ret

    def _parse_vasprun(self, vasprun):
        tree = ET.parse(vasprun)
        root = tree.getroot()

        calculation = []
        structures = []
        forces = []
        stresses = []
        orbital_magnetization = {}
        run_info = {}
        incar = {}
        general = {}
        kpoints_info = {}
        vasp_params = {}
        kpoints_list = []
        k_weights = []
        atom_info = {}
        for ichild in root:

            if ichild.tag == "generator":
                for ielement in ichild:
                    run_info[ielement.attrib["name"]] = ielement.text

            elif ichild.tag == "incar":
                incar = self.get_params(ichild, incar)

            # Skipping 1st structure which is primitive cell
            elif ichild.tag == "kpoints":
                for ielement in ichild:
                    tag_name = ielement.tag
                    element_items = ielement.items()
                    first_item_key = None
                    first_item_value = None
                    if len(element_items) > 0:
                        first_item_key = element_items[0][0]
                        first_item_value = element_items[0][1]
                        
                    if tag_name == "generation" or (first_item_key is not None and first_item_key == "param"):
                        kpoints_info["mode"] = first_item_value
                        if kpoints_info["mode"] == "listgenerated":
                            kpoints_info["kpoint_vertices"] = []
                            for isub in ielement:

                                if isub.attrib == "divisions":
                                    kpoints_info["ndivision"] = int(isub.text)
                                else:
                                    if len(isub.text.split()) != 3:
                                        continue
                                    kpoints_info["kpoint_vertices"].append(
                                        [float(x) for x in isub.text.split()]
                                    )
                        else:
                            for isub in ielement:
                                if isub.attrib["name"] == "divisions":
                                    kpoints_info["kgrid"] = [
                                        int(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "usershift":
                                    kpoints_info["user_shift"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec1":
                                    kpoints_info["genvec1"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec2":
                                    kpoints_info["genvec2"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec3":
                                    kpoints_info["genvec3"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "shift":
                                    kpoints_info["shift"] = [
                                        float(x) for x in isub.text.split()
                                    ]
         
                    elif tag_name == "varray" and first_item_key is not None and first_item_key == "kpointlist":
                        for ik in ielement:
                            kpoints_list.append([float(x) for x in ik.text.split()])
                        kpoints_list = np.array(kpoints_list)
                    elif tag_name == "varray" and first_item_key is not None and first_item_key == "weights":
                        for ik in ielement:
                            k_weights.append(float(ik.text))
                        k_weights = np.array(k_weights)
                        
                    elif tag_name == "kpoints_labels":
                        kpoints_info["kpoint_labels"] = []
                        for ik in ielement:
                            ik_items = ik.items()
                            kpoint_label = None
                            if len(ik_items) > 0:
                                kpoint_label = ik_items[0][1]
                            if kpoint_label is not None:
                                kpoints_info["kpoint_labels"].append(kpoint_label)

            # Vasp Parameters
            elif ichild.tag == "parameters":
                vasp_params = self.get_params(ichild, vasp_params)

            # Atom info
            elif ichild.tag == "atominfo":

                for ielement in ichild:
                    if ielement.tag == "atoms":
                        atom_info["natom"] = int(ielement.text)
                    elif ielement.tag == "types":
                        atom_info["nspecies"] = int(ielement.text)
                    elif ielement.tag == "array":
                        if ielement.attrib["name"] == "atoms":
                            for isub in ielement:
                                if isub.tag == "set":
                                    atom_info["symbols"] = []
                                    for isym in isub:
                                        atom_info["symbols"].append(isym[0].text)
                        elif ielement.attrib["name"] == "atomtypes":
                            atom_info["atom_types"] = {}
                            for isub in ielement:
                                if isub.tag == "set":
                                    for iatom in isub:
                                        atom_info["atom_types"][iatom[1].text] = {}
                                        atom_info["atom_types"][iatom[1].text][
                                            "natom_per_specie"
                                        ] = int(iatom[0].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "mass"
                                        ] = float(iatom[2].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "valance"
                                        ] = float(iatom[3].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "pseudopotential"
                                        ] = iatom[4].text.strip()

            elif ichild.tag == "structure":
                if ichild.attrib["name"] == "initialpos":
                    initial_pos = self.get_structure(ichild)
                elif ichild.attrib["name"] == "finalpos":
                    final_pos = self.get_structure(ichild)

            elif ichild.tag == "calculation":
                for ielement in ichild:
                    if ielement.tag == "scstep":
                        calculation.append(self.get_scstep(ielement))
                    elif ielement.tag == "structure":
                        structures.append(self.get_structure(ielement))
                    elif ielement.tag == "varray":
                        if ielement.attrib["name"] == "forces":
                            forces.append(self.get_varray(ielement))
                        elif ielement.attrib["name"] == "stress":
                            stresses.append(self.get_varray(ielement))

                    # elif ielement.tag == 'eigenvalues':
                    #     for isub in ielement[0] :
                    #         if isub.tag == 'set':
                    #             for iset in isub :
                    #                 eigen_values[iset.attrib['comment']] = {}
                    #                 for ikpt in iset :
                    #                     eigen_values[iset.attrib['comment']][ikpt.attrib['comment']] = get_varray(ikpt)

                    elif ielement.tag == "separator":
                        if ielement.attrib["name"] == "orbital magnetization":
                            for isub in ielement:
                                orbital_magnetization[isub.attrib["name"]] = [
                                    float(x) for x in isub.text.split()
                                ]

                    # elif ielement.tag == 'dos':
                    #     for isub in ielement :
                    #         if 'name' in isub.attrib:
                    #             if isub.attrib['name'] == 'fermi' :
                    #                 dos['fermi'] = float(isub.text)
                    #             else :
                    #                 dos[isub.tag] = {}
                    #                 dos[isub.tag]['info'] = []
                    #               for iset in isub[0]  :
                    #                   if iset.tag == 'set' :
                    #                       for isub_set in iset:
                    #                           dos[isub.tag] = get_set(isub_set,dos[isub.tag])
                    #                   elif iset.tag == 'field' :
                    #                       dos[isub.tag]['info'].append(iset.text.strip(' '))
                    else:
                        general[ielement.tag] = {}
                        general[ielement.tag] = self.get_general(
                            ielement, general[ielement.tag]
                        )
            # NEED TO ADD ORBITAL MAGNETIZATION

        return {
            "calculation": calculation,
            "structures": structures,
            "forces": forces,
            "run_info": run_info,
            "incar": incar,
            "general": general,
            "kpoints_info": kpoints_info,
            "vasp_params": vasp_params,
            "kpoints": {"kpoints_list": kpoints_list, "k_weights": k_weights},
            "atom_info": atom_info,
        }

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)