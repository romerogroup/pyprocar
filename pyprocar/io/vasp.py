# -*- coding: utf-8 -*-

from ..core import Structure, DensityOfStates, ElectronicBandStructure
import numpy as np
from numpy import array
import os
import re
import xml.etree.ElementTree as ET
import collections
import gzip


class VaspXML(collections.abc.Mapping):
    """contains."""

    def __init__(self, filename="vasprun.xml", dos_interpolation_factor=None):

        self.variables = {}
        self.dos_interpolation_factor = dos_interpolation_factor

        if not os.path.isfile(filename):
            raise ValueError("File not found " + filename)
        else:
            self.filename = filename

        self.spins_dict = {"spin 1": "Spin-up", "spin 2": "Spin-down"}
        # self.positions = None
        # self.stress = None
        # self.array_sizes = {}
        self.data = self.read()

    def read(self):
        """
        Read and parse vasprun.xml.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.parse_vasprun(self.filename)

    @property
    def bands(self):
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
                    bands_projected["projection"][
                        ispin, ikpoint, iband, :, :
                    ] = np.array(
                        self.data["general"]["projected"]["array"]["data"][spn][kpt][
                            kpt
                        ][bnd][bnd]
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

        spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
        energies = np.array(
            self.data["general"]["dos"]["total"]["array"]["data"][spins[0]]
        )[:, 0]
        dos_total = {"energies": energies}
        for ispin in spins:
            dos_total[self.spins_dict[ispin]] = np.array(
                self.data["general"]["dos"]["total"]["array"]["data"][ispin]
            )[:, 1]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self, atoms=[]):

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
            return (
                dos_projected,
                self.data["general"]["dos"]["partial"]["array"]["info"],
            )
        else:
            print("This calculation does not include partial density of states")
            return None, None

    @property
    def dos(self):
        energies = self.dos_total["energies"]
        total = []
        for ispin in self.dos_total:
            if ispin == "energies":
                continue
            total.append(self.dos_total[ispin])
        # total = np.array(total).T
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor,
        )

    @property
    def dos_to_dict(self): 
        """
        the complete density (total,projected) of states as a python dictionary
        """
        return {"total": self._get_dos_total(), "projected": self._get_dos_projected()}

    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """
        dos_total, labels = self._get_dos_total()
        dos_total["energies"] -= self.fermi

        return dos_total

    @property
    def dos_projected(self):
        """
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
            ret.append([temp_atom])
        return ret

    @property
    def kpoints(self):
        """
        Returns the kpoints used in the calculation in form of a pychemia.core.KPoints object
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
        Returns the list of kpoints and weights used in the calculation in form of a pychemia.core.KPoints object
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
        return self.data["general"]["dos"]["efermi"]

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
        """boolians in vaspxml are stores as T or F in str format, this function coverts them to python boolians """
        text = text.strip(" ")
        if text == "T" or text == ".True." or text == ".TRUE.":
            return True
        else:
            return False

    def conv(self, ele, _type):
        """This function converts the xml text to the type specified in the attrib of xml tree """

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
        """Returns an array for each varray tag in vaspxml """
        ret = []
        for ielement in xml_tree:
            ret.append([float(x) for x in ielement.text.split()])
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
        """Returns a dictionary of the structure """
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
        """This function extracts the self-consistent step information """
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
        """ This function will extract any element taged set recurcively"""
        if xml_tree[0].tag == "r":
            ret[xml_tree.attrib["comment"]] = self.get_varray(xml_tree)
            return ret
        else:
            ret[xml_tree.attrib["comment"]] = {}
            for ielement in xml_tree:

                if ielement.tag == "set":
                    ret[xml_tree.attrib["comment"]][ielement.attrib["comment"]] = {}
                    ret[xml_tree.attrib["comment"]][
                        ielement.attrib["comment"]
                    ] = self.get_set(
                        ielement,
                        ret[xml_tree.attrib["comment"]][ielement.attrib["comment"]],
                    )
            return ret

    def get_general(self, xml_tree, ret):
        """ This function will parse any element in calculatio other than the structures, scsteps"""
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
                        if ielement.attrib["name"] == "efermi":
                            ret["efermi"] = float(ielement.text)
                    continue
                ret[ielement.tag] = {}
                ret[ielement.tag] = self.get_general(ielement, ret[ielement.tag])
            return ret

    def parse_vasprun(self, vasprun):
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
                    if ielement.items()[0][0] == "param":
                        kpoints_info["mode"] = ielement.items()[0][1]
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

                    elif ielement.items()[0][1] == "kpointlist":
                        for ik in ielement:
                            kpoints_list.append([float(x) for x in ik.text.split()])
                        kpoints_list = array(kpoints_list)
                    elif ielement.items()[0][1] == "weights":
                        for ik in ielement:
                            k_weights.append(float(ik.text))
                        k_weights = array(k_weights)

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
                    #             if isub.attrib['name'] == 'efermi' :
                    #                 dos['efermi'] = float(isub.text)
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

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Poscar(Structure):
    def __init__(self, filename="CONTCAR"):
        self.filename = filename
        atoms, coordinates, lattice = self.parse_poscar()
        Structure.__init__(
            self, atoms=atoms, fractional_coordinates=coordinates, lattice=lattice
        )

    def parse_poscar(self):
        """
        Reads VASP POSCAR file-type and returns the pyprocar structure

        Parameters
        ----------
        filename : str, optional
            Path to POSCAR file. The default is 'CONTCAR'.

        Returns
        -------
        None.

        """
        rf = open(self.filename, "r")
        lines = rf.readlines()
        rf.close()
        comment = lines[0]
        self.comment = comment
        scale = float(lines[1])
        lattice = np.zeros(shape=(3, 3))
        for i in range(3):
            lattice[i, :] = [float(x) for x in lines[i + 2].split()[:3]]
        lattice *= scale
        if any([char.isalpha() for char in lines[5]]):
            species = [x for x in lines[5].split()]
            shift = 1
        else:
            shift = 0
            if os.path.exists("POTCAR"):
                base_dir = self.filename.replace(self.filename.split(os.sep)[-1], "")
                if base_dir == "":
                    base_dir = "."
                rf = open(base_dir + os.sep + "POTCAR", "r")
                potcar = rf.read()
                rf.close()
                species = re.findall(
                    "\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                    potcar,
                )[::2]
        composition = [int(x) for x in lines[5 + shift].split()]
        atoms = []
        for i in range(len(composition)):
            for x in composition[i] * [species[i]]:
                atoms.append(x)
        natom = sum(composition)
        if lines[6 + shift][0].lower() == "s":
            shift = 2
        if lines[6 + shift][0].lower() == "d":
            direct = True
        elif lines[6 + shift][0].lower() == "c":
            print("havn't implemented conversion to cartesian yet")
            direct = False
        coordinates = np.zeros(shape=(natom, 3))
        for i in range(natom):
            coordinates[i, :] = [float(x) for x in lines[i + 7 + shift].split()[:3]]

        if direct:
            return atoms, coordinates, lattice


class Procar(ElectronicBandStructure):
    def __init__(
        self,
        filename="PROCAR",
        structure=None,
        reciprocal_lattice=None,
        permissive=False,
        interpolation_factor=1,
    ):

        ElectronicBandStructure.__init__(
            self,
            structure=structure,
            reciprocal_lattice=reciprocal_lattice,
            interpolation_factor=interpolation_factor,
        )

        self.filename = filename
        self.permissive = permissive
        self.file_str = None
        self.data = self.file_str
        self.reciprocal_lattice = reciprocal_lattice
        
        self.orbitalName = [
            "s",
            "py",
            "pz",
            "px",
            "dxy",
            "dyz",
            "dz2",
            "dxz",
            "x2-y2",
            "fy3x2",
            "fxyz",
            "fyz2",
            "fz3",
            "fxz2",
            "fzx2",
            "fx3",
            "tot",
        ]
        self.orbitalName_old = [
            "s",
            "py",
            "pz",
            "px",
            "dxy",
            "dyz",
            "dz2",
            "dxz",
            "dx2",
            "tot",
        ]
        self.orbitalName_short = ["s", "p", "d", "f", "tot"]
        self.labels = self.orbitalName_old[:-1]

        self._read()
        self.eigen_values = self.bands
        self._spd2projected()

    def _open_file(self):
        """
        Tries to open a File, it has suitable values for PROCAR and can
        handle gzipped files

        Example:

            >>> foo =  UtilsProcar.Openfile()
            Tries to open "PROCAR", then "PROCAR.gz"

            >>> foo = UtilsProcar.Openfile("../bar")
            Tries to open "../bar". If it is a directory, it will try to open
            "../bar/PROCAR" and if fails again "../bar/PROCAR.gz"

            >>> foo = UtilsProcar.Openfile("PROCAR-spd.gz")
            Tries to open a gzipped file "PROCAR-spd.gz"

            If unable to open a file, it raises a "IOError" exception.
        """

        # checking if fileName is just a path and needs a "PROCAR to " be
        # appended
        if os.path.isdir(self.filename):
            if self.filename[-1] != r"/":
                self.filename += "/"
            self.filename += "PROCAR"

        # checking that the file exist
        if os.path.isfile(self.filename):
            # Checking if compressed
            if self.filename[-2:] == "gz":
                in_file = gzip.open(self.filename, mode="rt")
            else:
                in_file = open(self.filename, "r")
            return in_file

        # otherwise a gzipped version may exist
        elif os.path.isfile(self.filename + ".gz"):
            in_file = gzip.open(self.filename + ".gz", mode="rt")

        else:
            raise IOError("File not found")

        return in_file

    def _read(self):

        f = self._open_file()
        # Line 1: PROCAR lm decomposed
        if 'phase' in f.readline():
            self.has_phase = True
        # Line 2: # of k-points:  816   # of bands:  52   # of ions:   8
        metaLine = f.readline()  # metadata
        re.findall(r"#[^:]+:([^#]+)", metaLine)
        self.kpointsCount, self.bandsCount, self.ionsCount = map(
            int, re.findall(r"#[^:]+:([^#]+)", metaLine)
        )
        if self.ionsCount == 1:
            print(
                "Special case: only one atom found. The program may not work as expected"
            )
        else:
            self.ionsCount = self.ionsCount + 1

        # reading all the rest of the file to be parsed below
        self.file_str = f.read()
        self._readKpoints()
        self._readBands()
        self._readOrbital()
        return

    def _readKpoints(self):
        """Reads the k-point headers. A typical k-point line is:
        k-point    1 :    0.00000000 0.00000000 0.00000000  weight = 0.00003704\n
        fills self.kpoint[kpointsCount][3]
        The weights are discarded (are they useful?)
        """
        if not self.file_str:
            print("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all the K-points headers
        self.kpoints = re.findall(r"k-point\s+\d+\s*:\s+([-.\d\s]+)", self.file_str)

        # trying to build an array
        self.kpoints = [x.split() for x in self.kpoints]
        try:
            self.kpoints = np.array(self.kpoints, dtype=float)
        except ValueError:
            print("\n".join([str(x) for x in self.kpoints]))
            if self.permissive:
                # Discarding the kpoints list, however I need to set
                # self.ispin beforehand.
                if len(self.kpoints) == self.kpointsCount:
                    self.ispin = 1
                elif len(self.kpoints) == 2 * self.kpointsCount:
                    self.ispin = 2
                else:
                    raise ValueError("Kpoints do not match with ispin=1 or 2.")
                self.kpoints = None
                return
            else:
                raise ValueError("Badly formated Kpoints headers, try `--permissive`")
        # if successful, go on

        # trying to identify an non-polarized or non-collinear case, a
        # polarized case or a defective file

        if len(self.kpoints) != self.kpointsCount:
            # if they do not match, may means two things a spin polarized
            # case or a bad file, lets check
            # lets start testing if it is spin polarized, if so, there
            # should be 2 identical blocks of kpoints.
            up, down = np.vsplit(self.kpoints, 2)
            if (up == down).all():
                self.ispin = 2
                # just keeping one set of kpoints (the other will be
                # discarded)
                self.kpoints = up
            else:
                raise RuntimeError("Bad Kpoints list.")
        # if ISPIN != 2 setting ISPIN=1 (later for the non-collinear case 1->4)
        # It is unknown until parsing the projected data
        else:
            self.ispin = 1

        # checking again, for compatibility,
        if len(self.kpoints) != self.kpointsCount:
            raise RuntimeError(
                "Kpoints number do not match with metadata (header of PROCAR)"
            )

        if self.reciprocal_lattice is not None:
            self.kpoints = np.dot(self.kpoints, self.reciprocal_lattice)
        return

    def _readBands(self):
        """Reads the bands header. A typical bands is:
        band   1 # energy   -7.11986315 # occ.  1.00000000

        fills self.bands[kpointsCount][bandsCount]

        The occupation numbers are discarded (are they useful?)"""
        if not self.file_str:
            print("You should invoke `procar.read()` instead. Returning")
            return

        # finding all bands
        self.bands = re.findall(
            r"band\s*(\d+)\s*#\s*energy\s*([-.\d\s]+)", self.file_str
        )

        # checking if the number of bands match

        if len(self.bands) != self.bandsCount * self.kpointsCount * self.ispin:
            raise RuntimeError("Number of bands don't match")

        # casting to array to manipulate the bands
        self.bands = np.array(self.bands, dtype=float)

        # Now I will deal with the spin polarized case. The goal is join
        # them like for a non-magnetic case
        if self.ispin == 2:
            # up and down are along the first axis
            up, down = np.vsplit(self.bands, 2)

            # reshapping (the 2  means both band index and energy)
            up.shape = (self.kpointsCount, self.bandsCount, 2)
            down.shape = (self.kpointsCount, self.bandsCount, 2)

            # setting the correct number of bands (up+down)
            self.bandsCount *= 2

            # and joining along the second axis (axis=1), ie: bands-like
            self.bands = np.concatenate((up, down), axis=1)

        # otherwise just reshaping is needed
        else:
            self.bands.shape = (self.kpointsCount, self.bandsCount, 2)

        # Making a test if the broadcast is rigth, otherwise just print
        test = [x.max() - x.min() for x in self.bands[:, :, 0].transpose()]
        if np.array(test).any():
            print(
                "The indexes of bands do not match. CHECK IT. "
                "Likely the data was wrongly broadcasted"
            )
            print(str(self.bands[:, :, 0]))
        # Now safely removing the band index
        self.bands = self.bands[:, :, 1]
        return

    def _readOrbital(self):
        """Reads all the spd-projected data. A typical/expected block is:
            ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot
            1  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            2  0.152  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.152
            3  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            4  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            5  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            tot  0.686  0.000  0.002  0.000  0.000  0.000  0.000  0.000  0.000  0.688
            (x2 for spin-polarized -akwardkly formatted-, x4 non-collinear -nicely
             formatted-).

        The data is stored in an array self.spd[kpoint][band][ispin][atom][orbital]

        Undefined behavior in case of phase factors (LORBIT = 12).
        """
        if not self.file_str:
            print("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all orbital headers
        self.spd = re.findall(r"ion(.+)", self.file_str)

        # testing if the orbital names are known (the standard ones)
        FoundOrbs = self.spd[0].split()
        size = len(FoundOrbs)
        # only the first 'size' orbital
        StdOrbs = self.orbitalName[: size - 1] + self.orbitalName[-1:]
        StdOrbs_short = self.orbitalName_short[: size - 1] + self.orbitalName_short[-1:]
        StdOrbs_old = self.orbitalName_old[: size - 1] + self.orbitalName_old[-1:]
        if (
            FoundOrbs != (StdOrbs)
            and FoundOrbs != (StdOrbs_short)
            and FoundOrbs != (StdOrbs_old)
        ):
            print(
                str(size) + " orbitals. (Some of) They are unknow (if "
                "you did 'filter' them it is OK)."
            )
        self.orbitalCount = size
        self.orbitalNames = self.spd[0].split()

        # Now reading the bulk of data
        # The case of just one atom is handled differently since the VASP
        # output is a little different
        if self.ionsCount == 1:
            self.spd = re.findall(r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd = re.findall(r"ion.+tot\n([-.\d\seto]+)", self.file_str)
            self.spd = "".join(self.spd)
            self.spd = re.findall(r"([-.\d\se]+tot.+)\n", self.spd)
        # free the memory (could be a lot)
        self.file_str = None

        # Now the method will try to find the value of self.ispin,
        # previously it was set to either 1 or 2. If "1", it could be 1 or
        # 4, but previously it was impossible to find the rigth value. If
        # "2" it has to macth with the number of entries of spd data.

        expected = self.bandsCount * self.kpointsCount
        if expected == len(self.spd):
            pass
        # catching a non-collinear calc.
        elif expected * 4 == len(self.spd):
            # testing if previous ispin value is ok
            if self.ispin != 1:
                print(
                    "Incompatible data: self.ispin= " + str(self.ispin) + ". Now is 4"
                )
            self.ispin = 4
        else:
            raise RuntimeError("Shit happens")

        # checking for consistency
        for line in self.spd:
            if len(line.split()) != (self.ionsCount) * (self.orbitalCount + 1):
                raise RuntimeError("Flats happens")

        # replacing the "tot" string by a number, to allows a conversion
        # to numpy
        self.spd = [x.replace("tot", "0") for x in self.spd]
        self.spd = [x.split() for x in self.spd]
        self.spd = np.array(self.spd, dtype=float)

        # handling collinear polarized case
        if self.ispin == 2:
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(self.spd, 2)
            # ispin = 1 for a while, we will made the distinction
            up = up.reshape(
                self.kpointsCount,
                int(self.bandsCount / 2),
                1,
                self.ionsCount,
                self.orbitalCount + 1,
            )
            down = down.reshape(
                self.kpointsCount,
                int(self.bandsCount / 2),
                1,
                self.ionsCount,
                self.orbitalCount + 1,
            )
            # concatenating bandwise. Density and magntization, their
            # meaning is obvious, and do uses 2 times more memory than
            # required, but I *WANT* to keep it as close as possible to the
            # non-collinear or non-polarized case
            density = np.concatenate((up, down), axis=1)
            magnet = np.concatenate((up, -down), axis=1)
            # concatenated along 'ispin axis'
            self.spd = np.concatenate((density, magnet), axis=2)

        # otherwise, just a reshaping suffices
        else:
            self.spd.shape = (
                self.kpointsCount,
                self.bandsCount,
                self.ispin,
                self.ionsCount,
                self.orbitalCount + 1,
            )

        return
    
    
    # def readFile2(
    #     self,
    #     procar=None,
    #     phase=False,
    #     permissive=False,
    #     recLattice=None,
    #     ispin=None,  # the only spin channle to read
    # ):
    #     """
    #     Read file in a line by line manner.
    #     Only used when the phase factor is in procar. (for vasp, lorbit=12)
    #     """
    #     # Fall back to readFile function if no phase
    #     self.bands = None
    #     if not phase:
    #         self.readFile(
    #             procar=procar, phase=False, permissive=permissive, recLattice=recLattice
    #         )
    #     else:
    #         if ispin is None:
    #             nspin = 1
    #         else:
    #             nspin = 2
    #         iispin = 0
    #         self.projections = None
    #         ikpt = 0
    #         iband = 0
    #         nkread = 0
    #         # with open(self.fname) as myfile:
    #         f = self.utils.OpenFile(procar)
    #         lines = iter(f.readlines())
    #         last_iband = -1
    #         for line in lines:
    #             if line.startswith("# of k-points"):
    #                 a = re.findall(":\s*([0-9]*)", line)
    #                 self.kpointsCount, self.bandsCount, self.ionsCount = map(int, a)
    #                 self.kpoints = np.zeros([self.kpointsCount, 3])
    #                 self.kweights = np.zeros(self.kpointsCount)
    #                 if self.bands is None:
    #                     self.bands = np.zeros(
    #                         [nspin, self.kpointsCount, self.bandsCount]
    #                     )
    #             if line.strip().startswith("k-point"):
    #                 ss = line.strip().split()
    #                 ikpt = int(ss[1]) - 1
    #                 k0 = float(ss[3])
    #                 k1 = float(ss[4])
    #                 k2 = float(ss[5])
    #                 w = float(ss[-1])
    #                 self.kpoints[ikpt, :] = [k0, k1, k2]
    #                 self.kweights[ikpt] = w
    #                 nkread += 1
    #                 if nkread <= self.kpointsCount:
    #                     iispin = 0
    #                 else:
    #                     iispin = 1
    #             if line.strip().startswith("band"):
    #                 ss = line.strip().split()
    #                 try:
    #                     iband = int(ss[1]) - 1
    #                 except ValueError:
    #                     iband = last_iband + 1
    #                 last_iband = iband
    #                 e = float(ss[4])
    #                 occ = float(ss[-1])
    #                 self.bands[iispin, ikpt, iband] = e
    #             if line.strip().startswith("ion"):
    #                 if line.strip().endswith("tot"):
    #                     self.orbitalName = line.strip().split()[1:-1]
    #                     self.orbitalCount = len(self.orbitalName)
    #                 if self.projections is None:
    #                     self.projections = np.zeros(
    #                         [
    #                             self.kpointsCount,
    #                             self.bandsCount,
    #                             self.ionsCount,
    #                             self.orbitalCount,
    #                         ]
    #                     )
    #                     self.carray = np.zeros(
    #                         [
    #                             self.kpointsCount,
    #                             self.bandsCount,
    #                             nspin,
    #                             self.ionsCount,
    #                             self.orbitalCount,
    #                         ],
    #                         dtype="complex",
    #                     )
    #                 for i in range(self.ionsCount):
    #                     line = next(lines)
    #                     t = line.strip().split()
    #                     if len(t) == self.orbitalCount + 2:
    #                         self.projections[ikpt, iband, iispin, :] = [
    #                             float(x) for x in t[1:-1]
    #                         ]
    #                     elif len(t) == self.orbitalCount * 2 + 2:
    #                         self.carray[ikpt, iband, iispin, i, :] += np.array(
    #                             [float(x) for x in t[1:-1:2]]
    #                         )
    #                         self.carray[ikpt, iband, iispin, i, :] += 1j * np.array(
    #                             [float(x) for x in t[2::2]]
    #                         )

    #                     # Added by Francisco to parse older version of PROCAR format on Jun 11, 2019
    #                     elif len(t) == self.orbitalCount * 1 + 1:
    #                         self.carray[ikpt, iband, iispin, i, :] += np.array(
    #                             [float(x) for x in t[1:]]
    #                         )
    #                         line = next(lines)
    #                         t = line.strip().split()
    #                         self.carray[ikpt, iband, iispin, i, :] += 1j * np.array(
    #                             [float(x) for x in t[1:]]
    #                         )
    #                     else:
    #                         raise Exception(
    #                             "Cannot parse line to projection: %s" % line
    #                         )

