# -*- coding: utf-8 -*-

import numpy as np
from numpy import array
import os
import re
from ..core import Structure, DensityOfStates
import xml.etree.ElementTree as ET
import collections



class VaspXML(collections.abc.Mapping):
    """contains."""
    def __init__(self, filename='vasprun.xml', dos_interpolation_factor=None):

        self.variables = {}
        self.dos_interpolation_factor = dos_interpolation_factor

        if not os.path.isfile(filename):
            raise ValueError('File not found ' + filename)
        else:
            self.filename = filename

        self.spins_dict = {'spin 1': 'Spin-up', 'spin 2': 'Spin-down'}
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
        return parse_vasprun(self.filename)

    @property
    def bands(self):
        spins = list(
            self.data['general']['eigenvalues']['array']['data'].keys())
        kpoints_list = list(
            self.data['general']['eigenvalues']['array']['data']['spin 1'].keys())
        eigen_values = {}
        nbands = len(
            self.data['general']['eigenvalues']['array']['data'][spins[0]][kpoints_list[0]][kpoints_list[0]])
        nkpoints = len(kpoints_list)
        for ispin in spins:
            eigen_values[ispin] = {}
            eigen_values[ispin]['eigen_values'] = np.zeros(shape=(nbands,nkpoints))
            eigen_values[ispin]['occupancies'] = np.zeros(shape=(nbands,nkpoints))
            for ikpoint, kpt in enumerate(kpoints_list):
                temp = np.array(
                    self.data['general']['eigenvalues']['array']['data'][ispin][kpt][kpt])
                eigen_values[ispin]['eigen_values'][:,ikpoint] = temp[:, 0] - self.fermi
                eigen_values[ispin]['occupancies'][:,ikpoint] = temp[:, 1]
        return eigen_values
    
    @property
    def bands_projected(self):
        labels = self.data['general']['projected']['array']['info']
        spins = list(self.data['general']['projected']['array']['data'].keys())
        kpoints_list = list(self.data['general']['projected']['array']['data'][
            spins[0]].keys())
        bands_list = list(self.data['general']['projected']['array']['data'][
            spins[0]][kpoints_list[0]][kpoints_list[0]].keys())
        bands_projected = {'labels':labels}
        
        nkpoints = len(kpoints_list)
        nbands = len(bands_list)
        norbitals = len(labels)
        natoms = self.initial_structure.natoms
        for ispin in spins:
            bands_projected[ispin] = np.zeros(
                shape=(nkpoints, nbands, natoms, norbitals))
            for ikpoint, kpt in enumerate(kpoints_list):
                for iband, bnd in enumerate(bands_list):
                    bands_projected[ispin][ikpoint, iband, :, :] = np.array(
                        self.data['general']['projected']['array']['data'][
                            ispin][kpt][kpt][bnd][bnd])
        return bands_projected

    def _get_dos_total(self):

        spins = list(
            self.data['general']['dos']['total']['array']['data'].keys())
        energies = np.array(self.data['general']['dos']['total']['array']
                            ['data'][spins[0]])[:, 0]
        dos_total = {'energies': energies}
        for ispin in spins:
            dos_total[self.spins_dict[ispin]] = np.array(
                self.data['general']['dos']['total']['array']['data']
                [ispin])[:, 1]

        return dos_total, list(dos_total.keys())

        
    def _get_dos_projected(self, atoms=[]):

        if len(atoms) == 0:
            atoms = np.arange(self.initial_structure.natoms)

        if 'partial' in self.data['general']['dos']:
            dos_projected = {}
            ion_list = ["ion %s" % str(x + 1) for x in atoms
                        ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                name = self.initial_structure.atoms[atoms[i]] + str(atoms[i])
                spins = list(self.data['general']['dos']['partial']['array']
                             ['data'][iatom].keys())
                energies = np.array(
                    self.data['general']['dos']['partial']['array']['data']
                    [iatom][spins[0]][spins[0]])[:, 0]
                dos_projected[name] = {'energies': energies}
                for ispin in spins:
                    dos_projected[name][self.spins_dict[ispin]] = np.array(
                        self.data['general']['dos']['partial']['array']['data']
                        [iatom][ispin][ispin])[:, 1:]
            return dos_projected, self.data['general']['dos']['partial'][
                'array']['info']
        else:
            print(
                "This calculation does not include partial density of states")
            return None, None

    @property
    def dos(self):
        energies = self.dos_total['energies']
        total = []
        for ispin in self.dos_total:
            if ispin == 'energies':
                continue
            total.append(self.dos_total[ispin])
        # total = np.array(total).T
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor)

    @property
    def dos_to_dict(self):
        """
        Returns the complete density (total,projected) of states as a python dictionary
        """
        return {
            'total': self._get_dos_total(),
            'projected': self._get_dos_projected()
        }

    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """
        dos_total, labels = self._get_dos_total()
        dos_total['energies'] -= self.fermi

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
                    if key == 'energies':
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

        if self.data['kpoints_info']['mode'] == 'listgenerated':
            kpoints = dict(
                mode='path',
                kvertices=self.data['kpoints_info']['kpoint_vertices'])
        else:
            kpoints = dict(mode=self.data['kpoints_info']['mode'].lower(),
                           grid=self.data['kpoints_info']['kgrid'],
                           shifts=self.data['kpoints_info']['user_shift'])
        return kpoints

    @property
    def kpoints_list(self):
        """
        Returns the list of kpoints and weights used in the calculation in form of a pychemia.core.KPoints object
        """
        return dict(mode='reduced',
                    kpoints_list=self.data['kpoints']['kpoints_list'],
                    weights=self.data['kpoints']['k_weights'])

    @property
    def incar(self):
        """
        Returns the incar parameters used in the calculation as pychemia.code.vasp.VaspIncar object
        """
        return self.data['incar']

    @property
    def vasp_parameters(self):
        """
        Returns all of the parameters vasp has used in this calculation
        """
        return self.data['vasp_params']

    @property
    def potcar_info(self):
        """
        Returns the information about pseudopotentials(POTCAR) used in this calculation
        """
        return self.data['atom_info']['atom_types']

    @property
    def fermi(self):
        """
        Returns the fermi energy
        """
        return self.data['general']['dos']['efermi']

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
        symbols = [x.strip() for x in self.data['atom_info']['symbols']]
        structures = []
        for ist in self.data['structures']:

            st = Structure(atoms=symbols,
                           fractional_coordinates=ist['reduced'],
                           lattice=ist['cell'])
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
        return self.data['forces']

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
        return self.data['calculation']

    @property
    def energies(self):
        """
        Returns a list of energies in each electronic and ionic step [ionic step,electronic step, energy]
        """
        scf_step = 0
        ion_step = 0
        double_counter = 1
        energies = []
        for calc in self.data['calculation']:
            if 'ewald' in calc['energy']:
                if double_counter == 0:
                    double_counter += 1
                    scf_step += 1
                elif double_counter == 1:
                    double_counter = 0
                    ion_step += 1
                    scf_step = 1
            else:
                scf_step += 1
            energies.append([ion_step, scf_step, calc['energy']['e_0_energy']])
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
        ediff = self.vasp_parameters['electronic']['EDIFF']
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
            print('This calculation does not have ionic steps')
            return True
        else:
            ediffg = self.vasp_parameters['ionic']['EDIFFG']
            if ediffg < 0:
                last_forces_abs = np.abs(np.array(self.forces[-1]))
                return not (np.any(last_forces_abs > abs(ediffg)))
            else:
                last_ionic_energy = energies[(
                    energies[:, 0] == nsteps)][-1][-1]
                penultimate_ionic_energy = energies[(energies[:, 0] == (
                    nsteps - 1))][-1][-1]
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
        return (self.convergence_electronic and self.convergence_ionic)

    @property
    def is_finished(self):
        """
        Always returns True, need to fix this according to reading the xml as if the calc is
        not finished we will have errors in xml parser
        """
        # if vasprun.xml is read the calculation is finished
        return True

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


def text_to_bool(text):
    """boolians in vaspxml are stores as T or F in str format, this function coverts them to python boolians """
    text = text.strip(' ')
    if text == 'T' or text == '.True.' or text == '.TRUE.':
        return True
    else:
        return False


def conv(ele, _type):
    """This function converts the xml text to the type specified in the attrib of xml tree """

    if _type == 'string':
        return ele.strip()
    elif _type == 'int':
        return int(ele)
    elif _type == 'logical':
        return text_to_bool(ele)
    elif _type == 'float':
        if '*' in ele:
            return np.nan
        else:
            return float(ele)


def get_varray(xml_tree):
    """Returns an array for each varray tag in vaspxml """
    ret = []
    for ielement in xml_tree:
        ret.append([float(x) for x in ielement.text.split()])
    return ret


def get_params(xml_tree, dest):
    """dest should be a dictionary
    This function is recurcive #check spelling"""
    for ielement in xml_tree:
        if ielement.tag == 'separator':
            dest[ielement.attrib['name'].strip()] = {}
            dest[ielement.attrib['name'].strip()] = get_params(
                ielement, dest[ielement.attrib['name']])
        else:
            if 'type' in ielement.attrib:
                _type = ielement.attrib['type']
            else:
                _type = 'float'
            if ielement.text is None:
                dest[ielement.attrib['name'].strip()] = None

            elif len(ielement.text.split()) > 1:
                dest[ielement.attrib['name'].strip()] = [
                    conv(x, _type) for x in ielement.text.split()
                ]
            else:
                dest[ielement.attrib['name'].strip()] = conv(
                    ielement.text, _type)

    return dest


def get_structure(xml_tree):
    """Returns a dictionary of the structure """
    ret = {}
    for ielement in xml_tree:
        if ielement.tag == 'crystal':
            for isub in ielement:
                if isub.attrib['name'] == 'basis':
                    ret['cell'] = get_varray(isub)
                elif isub.attrib['name'] == 'volume':
                    ret['volume'] = float(isub.text)
                elif isub.attrib['name'] == 'rec_basis':
                    ret['rec_cell'] = get_varray(isub)
        elif ielement.tag == 'varray':
            if ielement.attrib['name'] == 'positions':
                ret['reduced'] = get_varray(ielement)
    return ret


def get_scstep(xml_tree):
    """This function extracts the self-consistent step information """
    scstep = {'time': {}, 'energy': {}}
    for isub in xml_tree:
        if isub.tag == 'time':
            scstep['time'][isub.attrib['name']] = [
                float(x) for x in isub.text.split()
            ]
        elif isub.tag == 'energy':
            for ienergy in isub:
                scstep['energy'][ienergy.attrib['name']] = float(ienergy.text)
    return scstep


def get_set(xml_tree, ret):
    """ This function will extract any element taged set recurcively"""
    if xml_tree[0].tag == 'r':
        ret[xml_tree.attrib['comment']] = get_varray(xml_tree)
        return ret
    else:
        ret[xml_tree.attrib['comment']] = {}
        for ielement in xml_tree:

            if ielement.tag == 'set':
                ret[xml_tree.attrib['comment']][
                    ielement.attrib['comment']] = {}
                ret[xml_tree.attrib['comment']][
                    ielement.attrib['comment']] = get_set(
                        ielement, ret[xml_tree.attrib['comment']][
                            ielement.attrib['comment']])
        return ret


def get_general(xml_tree, ret):
    """ This function will parse any element in calculatio other than the structures, scsteps"""
    if 'dimension' in [x.tag for x in xml_tree]:
        ret['info'] = []
        ret['data'] = {}
        for ielement in xml_tree:
            if ielement.tag == 'field':
                ret['info'].append(ielement.text.strip(' '))
            elif ielement.tag == 'set':
                for iset in ielement:
                    ret['data'] = get_set(iset, ret['data'])
        return ret
    else:
        for ielement in xml_tree:
            if ielement.tag == 'i':
                if 'name' in ielement.attrib:
                    if ielement.attrib['name'] == 'efermi':
                        ret['efermi'] = float(ielement.text)
                continue
            ret[ielement.tag] = {}
            ret[ielement.tag] = get_general(ielement, ret[ielement.tag])
        return ret


def parse_vasprun(vasprun):
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

        if ichild.tag == 'generator':
            for ielement in ichild:
                run_info[ielement.attrib['name']] = ielement.text

        elif ichild.tag == 'incar':
            incar = get_params(ichild, incar)

        # Skipping 1st structure which is primitive cell
        elif ichild.tag == 'kpoints':

            for ielement in ichild:
                if ielement.items()[0][0] == 'param':
                    kpoints_info['mode'] = ielement.items()[0][1]
                    if kpoints_info['mode'] == 'listgenerated':
                        kpoints_info['kpoint_vertices'] = []
                        for isub in ielement:

                            if isub.attrib == 'divisions':
                                kpoints_info['ndivision'] = int(isub.text)
                            else:
                                if len(isub.text.split()) != 3:
                                    continue
                                kpoints_info['kpoint_vertices'].append(
                                    [float(x) for x in isub.text.split()])
                    else:
                        for isub in ielement:
                            if isub.attrib['name'] == 'divisions':
                                kpoints_info['kgrid'] = [
                                    int(x) for x in isub.text.split()
                                ]
                            elif isub.attrib['name'] == 'usershift':
                                kpoints_info['user_shift'] = [
                                    float(x) for x in isub.text.split()
                                ]
                            elif isub.attrib['name'] == 'genvec1':
                                kpoints_info['genvec1'] = [
                                    float(x) for x in isub.text.split()
                                ]
                            elif isub.attrib['name'] == 'genvec2':
                                kpoints_info['genvec2'] = [
                                    float(x) for x in isub.text.split()
                                ]
                            elif isub.attrib['name'] == 'genvec3':
                                kpoints_info['genvec3'] = [
                                    float(x) for x in isub.text.split()
                                ]
                            elif isub.attrib['name'] == 'shift':
                                kpoints_info['shift'] = [
                                    float(x) for x in isub.text.split()
                                ]

                elif ielement.items()[0][1] == 'kpointlist':
                    for ik in ielement:
                        kpoints_list.append(
                            [float(x) for x in ik.text.split()])
                    kpoints_list = array(kpoints_list)
                elif ielement.items()[0][1] == 'weights':
                    for ik in ielement:
                        k_weights.append(float(ik.text))
                    k_weights = array(k_weights)

        # Vasp Parameters
        elif ichild.tag == 'parameters':
            vasp_params = get_params(ichild, vasp_params)

        # Atom info
        elif ichild.tag == 'atominfo':

            for ielement in ichild:
                if ielement.tag == 'atoms':
                    atom_info['natom'] = int(ielement.text)
                elif ielement.tag == 'types':
                    atom_info['nspecies'] = int(ielement.text)
                elif ielement.tag == 'array':
                    if ielement.attrib['name'] == 'atoms':
                        for isub in ielement:
                            if isub.tag == 'set':
                                atom_info['symbols'] = []
                                for isym in isub:
                                    atom_info['symbols'].append(isym[0].text)
                    elif ielement.attrib['name'] == 'atomtypes':
                        atom_info['atom_types'] = {}
                        for isub in ielement:
                            if isub.tag == 'set':
                                for iatom in isub:
                                    atom_info['atom_types'][iatom[1].text] = {}
                                    atom_info['atom_types'][iatom[1].text][
                                        'natom_per_specie'] = int(
                                            iatom[0].text)
                                    atom_info['atom_types'][
                                        iatom[1].text]['mass'] = float(
                                            iatom[2].text)
                                    atom_info['atom_types'][
                                        iatom[1].text]['valance'] = float(
                                            iatom[3].text)
                                    atom_info['atom_types'][iatom[1].text][
                                        'pseudopotential'] = iatom[
                                            4].text.strip()

        elif ichild.tag == 'structure':
            if ichild.attrib['name'] == 'initialpos':
                initial_pos = get_structure(ichild)
            elif ichild.attrib['name'] == 'finalpos':
                final_pos = get_structure(ichild)

        elif ichild.tag == 'calculation':
            for ielement in ichild:
                if ielement.tag == 'scstep':
                    calculation.append(get_scstep(ielement))
                elif ielement.tag == 'structure':
                    structures.append(get_structure(ielement))
                elif ielement.tag == 'varray':
                    if ielement.attrib['name'] == 'forces':
                        forces.append(get_varray(ielement))
                    elif ielement.attrib['name'] == 'stress':
                        stresses.append(get_varray(ielement))

                # elif ielement.tag == 'eigenvalues':
                #     for isub in ielement[0] :
                #         if isub.tag == 'set':
                #             for iset in isub :
                #                 eigen_values[iset.attrib['comment']] = {}
                #                 for ikpt in iset :
                #                     eigen_values[iset.attrib['comment']][ikpt.attrib['comment']] = get_varray(ikpt)

                elif ielement.tag == 'separator':
                    if ielement.attrib['name'] == "orbital magnetization":
                        for isub in ielement:
                            orbital_magnetization[isub.attrib['name']] = [
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
                    general[ielement.tag] = get_general(
                        ielement, general[ielement.tag])
        # NEED TO ADD ORBITAL MAGNETIZATION

    return {
        'calculation': calculation,
        'structures': structures,
        'forces': forces,
        'run_info': run_info,
        'incar': incar,
        'general': general,
        'kpoints_info': kpoints_info,
        'vasp_params': vasp_params,
        'kpoints': {
            'kpoints_list': kpoints_list,
            'k_weights': k_weights
        },
        'atom_info': atom_info
    }



def parse_poscar(filename='CONTCAR'):
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
    rf = open(filename,'r')
    lines = rf.readlines()
    rf.close()
    comment = lines[0]
    scale = float(lines[1])
    lattice = np.zeros(shape=(3, 3))
    for i in range(3):
        lattice[i, :] = [float(x) for x in lines[i+2].split()[:3]]
    lattice *= scale
    if any([char.isalpha() for char in lines[5]]):
        species = [x for x in lines[5].split()]
        shift = 1
    else :
        shift = 0
        if os.path.exists('POTCAR'):
            base_dir = filename.replace(filename.split(os.sep)[-1], "")
            if base_dir=='':
                base_dir='.'
            rf = open(base_dir+os.sep+'POTCAR','r')
            potcar = rf.read()
            rf.close()
            species = re.findall("\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                       potcar)[::2]
    composition = [int(x) for x in lines[5+shift].split()]
    atoms = []
    for i in range(len(composition)):
        for x in composition[i]*[species[i]]:
            atoms.append(x)
    natom = sum(composition)
    if lines[6+shift][0].lower() == 's':
        shift = 2
    if lines[6+shift][0].lower() == 'd':
        direct = True
    elif lines[6+shift][0].lower() == 'c':
        print("havn't implemented conversion to cartesian yet")
        direct = False
    coordinates = np.zeros(shape=(natom, 3))
    for i in range(natom):
        coordinates[i,:] = [float(x) for x in lines[i+7+shift].split()[:3]]
    if direct :
        return Structure(atoms=atoms, 
                         fractional_coordinates=coordinates, 
                         lattice=lattice)
    else:
        return Structure(atoms=atoms, 
                         cartesian_coordinates=coordinates, 
                         lattice=lattice)
    
        
    
    
    
    