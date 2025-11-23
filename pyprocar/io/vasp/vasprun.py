import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np
from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class GeneratorInfo:
    """Generator information from vasprun.xml"""
    program: str
    version: str
    subversion: str
    platform: str
    date: str
    time: str


@dataclass
class KPointsInfo:
    """KPoints information from vasprun.xml"""
    weights: np.ndarray
    kpointlist: np.ndarray
    comment: str
    mode: str
    ngrids: int
    automatic: bool
    kgrid: list[int]
    kshift: list[float]


@dataclass
class CrystalInfo:
    """Crystal information from vasprun.xml"""
    basis: np.ndarray
    volume: float
    rec_basis: np.ndarray


@dataclass
class StructureInfo:
    """Structure information from vasprun.xml"""
    crystal: CrystalInfo
    positions: np.ndarray


@dataclass
class TimeInfo:
    """Timing information for SCF steps"""
    dav: float
    total: float


@dataclass
class EnergyInfo:
    """Energy information for SCF steps"""
    e_fr_energy: float
    e_wo_entrp: float
    e_0_energy: float
    alphaZ: float | None = None
    ewald: float | None = None
    hartreedc: float | None = None
    XCdc: float | None = None
    pawpsdc: float | None = None
    pawaedc: float | None = None
    eentropy: float | None = None
    bandstr: float | None = None
    atom: float | None = None


@dataclass
class SCStepInfo:
    """Self-consistent step information"""
    time: TimeInfo
    energy: EnergyInfo


def parse_parameters_child(element: etree._Element, separator_name: str) -> dict[str, Any]:
    assert element is not None
    
    tag_list = element.xpath(f"//separator[@name='{separator_name}']")
    assert isinstance(tag_list, Iterable)
    if not tag_list:
        return {}
    
    sep_elem = tag_list[0]
    if not isinstance(sep_elem, etree._Element):
        return {}
    
    params: dict[str, Any] = {}
    for subelement in sep_elem:
        tag_name = subelement.tag
        name = subelement.attrib.get("name", None)
        _type = subelement.attrib.get("type", None)
        text = subelement.text
        assert isinstance(text, str)
        
        if tag_name == "separator":
            continue
        elif tag_name == "v":
            if _type == "int":
                value = np.asarray(text.strip().split(), dtype=int)
            else:
                value = np.asarray(text.strip().split(), dtype=float)
        elif _type == "logical":
            value = text.strip() == "T"
        elif _type == "int":
            value = int(text.strip())
        elif _type == "string":
            value = text.strip()
        elif tag_name == "i" and _type is None:
            try:
                value = float(text.strip())
            except ValueError:
                value = text.strip()
        else:
            logger.warning(f"The vasprun.xml parameter {tag_name} with type {_type} is not valid")
            continue
        assert name is not None, "The vasprun.xml parameter is not valid"
        params[name.strip()] = value
    return params


class VaspXML(Mapping[str, Any]):
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

    non_colinear_spins_dict: dict[str, str] = {
        "spin 1": "Spin-Total",
        "spin 2": "Spin-x",
        "spin 3": "Spin-y",
        "spin 4": "Spin-z",
    }
    colinear_spins_dict: dict[str, str] = {"spin 1": "Spin-up", "spin 2": "Spin-down"}

    def __init__(self, filepath: str | Path | None = "vasprun.xml", file_str: str = ""):

        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str
        self._etree: etree._ElementTree | None = None

    @classmethod
    def from_str(cls, input: str):
        return cls(filepath=None, file_str=input)
    
    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)
    
    @cached_property
    def filename(self) -> str:
        if self.filepath is not None:
            return self.filepath.name
        return ""
    
    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as xml:
                return xml.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @cached_property
    def root(self) -> etree._Element:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as f:
                self._file_str = f.read()
                
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        xml_str = re.sub(r'<\?xml[^>]+\?>', '', self._file_str)
        return etree.fromstring(xml_str)

#     @cached_property
#     def data(self) -> dict[str, Any]:
#         return self._parse_vasprun()

    @property
    def has_dos(self) -> bool:
        return self.dos_element is not None
    
    @property
    def dos_element(self) -> etree._Element | None:
        # for chil in self.root
        path_list = self.root.xpath("//dos")
        assert isinstance(path_list, Iterable)
        
        element = path_list[0] if path_list else None
        
        assert isinstance(element, etree._Element)
        return element[0] if element else None

#     @property
#     def spins_dict(self) -> dict[str, str]:

#         spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
#         if len(spins) == 4:
#             return self.non_colinear_spins_dict
#         else:
#             return self.colinear_spins_dict

    
    @cached_property
    def parameters_element(self) -> etree._Element | None:
        path_list = self.root.xpath("//parameters")
        assert isinstance(path_list, Iterable)
        
        element = path_list[0] if path_list else None
        
        if element is not None:
            assert isinstance(element, etree._Element)
            return element[0] if len(element) > 0 else None
        return None
    
    @cached_property
    def general_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "general")
    
    @cached_property
    def electronic_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic")
    
    @cached_property
    def electronic_smearing_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic smearing")
    
    @cached_property
    def electronic_projectors_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic projectors")
    
    @cached_property
    def electronic_startup_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic startup")
    
    @cached_property
    def electronic_spin_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic spin")
    
    @cached_property
    def electronic_exchange_correlation_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic exchange-correlation")
    
    @cached_property
    def electronic_convergence_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic convergence")
    
    @cached_property
    def electronic_convergence_detail_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic convergence detail")
    
    @cached_property
    def electronic_mixer_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic mixer")
    
    @cached_property
    def electronic_mixer_details_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic mixer details")
    
    @cached_property
    def electronic_dipolcorrection_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "electronic dipolcorrection")
    
    @cached_property
    def grids_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "grids")
    
    @cached_property
    def ionic_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "ionic")
    
    @cached_property
    def ionic_md_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "ionic md")
    
    @cached_property
    def symmetry_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "symmetry")
    
    @cached_property
    def dos_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "dos")
    
    @cached_property
    def writing_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "writing")
    
    @cached_property
    def performance_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "performance")
    
    @cached_property
    def miscellaneous_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "miscellaneous")
    
    @cached_property
    def ldau_parameters(self) -> dict[str, Any]:
        assert self.parameters_element is not None
        params: dict[str, Any] = {}
        tag_list = self.root.xpath("//parameters")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return {}
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        for subelement in element:
            if subelement.tag == "separator":
                continue
            name = subelement.attrib.get("name", None)
            _type = subelement.attrib.get("type", None)
            text = subelement.text
            
            if name not in ["GGA_COMPAT", "LBERRY", "ICORELEVEL", "LDAU", "LDAUTYPE", "LDAUL", 
                            "LDAUU", "LDAUJ", "LDAUPRINT", "I_CONSTRAINED_M"]:
                continue
            
            if text is None:
                continue
            
            if _type == "logical":
                value = text.strip() == "T"
            elif _type == "int":
                value = int(text.strip())
            elif subelement.tag == "v":
                if _type == "int":
                    value = np.asarray(text.strip().split(), dtype=int)
                else:
                    value = np.asarray(text.strip().split(), dtype=float)
            else:
                try:
                    value = float(text.strip())
                except ValueError:
                    value = text.strip()
            
            params[name] = value
        return params
    
    @cached_property
    def exchange_correlation_parameters(self) -> dict[str, Any]:
        assert self.parameters_element is not None
        tag_list = self.root.xpath("//parameters/separator"+
                                   "[@name='electronic exchange-correlation']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return {}
        element = tag_list[-1]
        assert isinstance(element, etree._Element)
        params: dict[str , Any] = {}
        for subelement in element:
            name = subelement.attrib.get("name", None)
            _type = subelement.attrib.get("type", None)
            text = subelement.text
            if name is None:
                continue
            name = name.strip()
            if text is None:
                if _type == "string":
                    value = ""
                else:
                    continue
            elif _type == "logical":
                value = text.strip() == "T"
            elif _type == "int":
                value = int(text.strip())
            elif _type == "string":
                value = text.strip()
            else:
                try:
                    value = float(text.strip())
                except ValueError:
                    value = text.strip()
            params[name] = value
        return params
    
    @cached_property
    def vdw_dft_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "vdW DFT")
    
    @cached_property
    def model_gw_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "model GW")
    
    @cached_property
    def linear_response_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "linear response parameters")
    
    @cached_property
    def orbital_magnetization_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "orbital magnetization")
    
    @cached_property
    def response_functions_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "response functions")
    
    @cached_property
    def external_order_field_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "External order field")
    
    @cached_property
    def optional_k_points_parameters(self) -> dict[str, Any] | None:
        assert self.parameters_element is not None
        return parse_parameters_child(self.parameters_element, "optional k-points parameters")
    
    @cached_property
    def incar_parameters(self) -> dict[str, Any] | None:
        """Parse INCAR parameters from vasprun.xml"""
        tag_list = self.root.xpath("//incar")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        params: dict[str, Any] = {}
        for subelement in element:
            name = subelement.attrib.get("name", None)
            _type = subelement.attrib.get("type", None)
            text = subelement.text
            
            if name is None or text is None:
                continue
            
            name = name.strip()
            
            if subelement.tag == "v":
                # Vector data
                if _type == "int":
                    value = [int(x) for x in text.strip().split()]
                else:
                    value = [float(x) for x in text.strip().split()]
            elif _type == "logical":
                value = text.strip() == "T"
            elif _type == "int":
                value = int(text.strip())
            elif _type == "string":
                value = text.strip()
            else:
                # Default to float
                try:
                    value = float(text.strip())
                except ValueError:
                    value = text.strip()
            
            params[name] = value
        
        return params
    
    @cached_property
    def generator_parameters(self) -> GeneratorInfo | None:
        """Parse generator information from vasprun.xml"""
        tag_list = self.root.xpath("//generator")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        info: dict[str, str] = {}
        for subelement in element:
            name = subelement.attrib.get("name", None)
            text = subelement.text
            
            if name is not None and text is not None:
                info[name] = text.strip()
        
        return GeneratorInfo(
            program=info.get("program", ""),
            version=info.get("version", ""),
            subversion=info.get("subversion", ""),
            platform=info.get("platform", ""),
            date=info.get("date", ""),
            time=info.get("time", "")
        )
    
    @cached_property
    def kpoints(self) -> KPointsInfo | None:
        """Parse kpoints information from vasprun.xml"""
        tag_list = self.root.xpath("//kpoints")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        weights_list: list[float] = []
        kpointlist_data: list[list[float]] = []
        mode = ""
        ngrids = 0
        kgrid = [2, 2, 2]  # default
        kshift = [0.0, 0.0, 0.0]  # default
        
        for subelement in element:
            if subelement.tag == "generation":
                mode = subelement.attrib.get("param", "")
                for subsubelement in subelement:
                    if subsubelement.attrib.get("name") == "divisions":
                        if subsubelement.text:
                            ngrids = int(subsubelement.text.strip())
            elif subelement.tag == "varray":
                name = subelement.attrib.get("name", "")
                if name == "kpointlist":
                    for v_elem in subelement:
                        if v_elem.text:
                            kpointlist_data.append([float(x) for x in v_elem.text.split()])
                elif name == "weights":
                    for v_elem in subelement:
                        if v_elem.text:
                            weights_list.append(float(v_elem.text.strip()))
        
        weights = np.array(weights_list)
        kpointlist = np.array(kpointlist_data)
        automatic = False  # Based on mode
        
        return KPointsInfo(
            weights=weights,
            kpointlist=kpointlist,
            comment=mode,
            mode=mode,
            ngrids=ngrids,
            automatic=automatic,
            kgrid=kgrid,
            kshift=kshift
        )
    
    @cached_property
    def eigenvalues(self) -> np.ndarray | None:
        """Parse eigenvalues from vasprun.xml
        
        Returns array with shape (n_bands, n_kpoints, n_spins)
        """
        tag_list = self.root.xpath("//eigenvalues")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Find the array element
        array_elem = None
        for child in element:
            if child.tag == "array":
                array_elem = child
                break
        
        if array_elem is None:
            return None
        
        # Find the set element containing spin sets
        set_elem = None
        for child in array_elem:
            if child.tag == "set":
                set_elem = child
                break
        
        if set_elem is None:
            return None
        
        # Parse spin sets
        spin_data: list[list[list[float]]] = []
        for spin_set in set_elem:
            if spin_set.tag == "set":
                kpoint_data: list[list[float]] = []
                for kpoint_set in spin_set:
                    if kpoint_set.tag == "set":
                        band_data: list[float] = []
                        for r_elem in kpoint_set:
                            if r_elem.tag == "r" and r_elem.text:
                                values = r_elem.text.split()
                                band_data.append(float(values[0]))  # eigenvalue
                        kpoint_data.append(band_data)
                spin_data.append(kpoint_data)
        
        # Convert to numpy array and transpose to (n_bands, n_kpoints, n_spins)
        arr = np.array(spin_data)  # shape: (n_spins, n_kpoints, n_bands)
        arr = np.transpose(arr, (2, 1, 0))  # shape: (n_bands, n_kpoints, n_spins)
        
        return arr
    
    @cached_property
    def total(self) -> np.ndarray | None:
        """Parse total DOS from vasprun.xml
        
        Returns array with shape (n_energies, n_spins)
        """
        tag_list = self.root.xpath("//total")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Find the array element
        array_elem = None
        for child in element:
            if child.tag == "array":
                array_elem = child
                break
        
        if array_elem is None:
            return None
        
        # Find the set element
        set_elem = None
        for child in array_elem:
            if child.tag == "set":
                set_elem = child
                break
        
        if set_elem is None:
            return None
        
        # Parse spin sets
        spin_data: list[list[float]] = []
        for spin_set in set_elem:
            if spin_set.tag == "set":
                energy_data: list[float] = []
                for r_elem in spin_set:
                    if r_elem.tag == "r" and r_elem.text:
                        values = r_elem.text.split()
                        energy_data.append(float(values[1]))  # total DOS (skip energy)
                spin_data.append(energy_data)
        
        # Convert to numpy array and transpose to (n_energies, n_spins)
        arr = np.array(spin_data)  # shape: (n_spins, n_energies)
        arr = np.transpose(arr, (1, 0))  # shape: (n_energies, n_spins)
        
        return arr
    
    @cached_property
    def partial(self) -> np.ndarray | None:
        """Parse partial DOS from vasprun.xml
        
        Returns array with shape (n_energies, n_spins, n_ions, n_orbitals)
        """
        tag_list = self.root.xpath("//partial")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Find the array element
        array_elem = None
        for child in element:
            if child.tag == "array":
                array_elem = child
                break
        
        if array_elem is None:
            return None
        
        # Find the set element
        set_elem = None
        for child in array_elem:
            if child.tag == "set":
                set_elem = child
                break
        
        if set_elem is None:
            return None
        
        # Parse ion sets
        ion_data: list[list[list[list[float]]]] = []
        for ion_set in set_elem:
            if ion_set.tag == "set":
                spin_data: list[list[list[float]]] = []
                for spin_set in ion_set:
                    if spin_set.tag == "set":
                        energy_data: list[list[float]] = []
                        for r_elem in spin_set:
                            if r_elem.tag == "r" and r_elem.text:
                                values = r_elem.text.split()
                                # Skip first value (energy), take rest as orbital projections
                                orbitals = [float(v) for v in values[1:]]
                                energy_data.append(orbitals)
                        spin_data.append(energy_data)
                ion_data.append(spin_data)
        
        # Convert to numpy array: (n_ions, n_spins, n_energies, n_orbitals)
        arr = np.array(ion_data)
        # Transpose to (n_energies, n_spins, n_ions, n_orbitals)
        arr = np.transpose(arr, (2, 1, 0, 3))
        
        return arr
    
    @cached_property
    def projected(self) -> np.ndarray | None:
        """Parse projected eigenvalues from vasprun.xml
        
        Returns array with shape (n_bands, n_kpoints, n_spins, n_ions, n_orbitals)
        """
        tag_list = self.root.xpath("//projected")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Find the array elements - second one is the projection data
        array_elems = [child for child in element if child.tag == "array"]
        if len(array_elems) < 2:
            # If there's only one array, use it
            array_elem = array_elems[0] if array_elems else None
        else:
            # Use the second array (first is eigenvalues)
            array_elem = array_elems[1]
        
        if array_elem is None:
            return None
        
        # Find the set element
        set_elem = None
        for child in array_elem:
            if child.tag == "set":
                set_elem = child
                break
        
        if set_elem is None:
            return None
        
        # Parse spin sets -> kpoint sets -> band sets -> ion rows
        spin_data: list[list[list[list[list[float]]]]] = []
        for spin_set in set_elem:
            if spin_set.tag == "set":
                kpoint_data: list[list[list[list[float]]]] = []
                for kpoint_set in spin_set:
                    if kpoint_set.tag == "set":
                        band_data: list[list[list[float]]] = []
                        for band_set in kpoint_set:
                            if band_set.tag == "set":
                                ion_data: list[list[float]] = []
                                for r_elem in band_set:
                                    if r_elem.tag == "r" and r_elem.text:
                                        values = r_elem.text.split()
                                        orbitals = [float(v) for v in values]
                                        ion_data.append(orbitals)
                                band_data.append(ion_data)
                        kpoint_data.append(band_data)
                spin_data.append(kpoint_data)
        
        # Convert to numpy array: (n_spins, n_kpoints, n_bands, n_ions, n_orbitals)
        arr = np.array(spin_data)
        # Transpose to (n_bands, n_kpoints, n_spins, n_ions, n_orbitals)
        arr = np.transpose(arr, (2, 1, 0, 3, 4))
        
        return arr
    
    @cached_property
    def forces(self) -> np.ndarray | None:
        """Parse forces from vasprun.xml
        
        Returns array with shape (n_ions, 3)
        """
        tag_list = self.root.xpath("//calculation/forces/varray[@name='forces']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Parse force vectors
        force_data: list[list[float]] = []
        for v_elem in element:
            if v_elem.tag == "v" and v_elem.text:
                values = v_elem.text.split()
                force_data.append([float(v) for v in values])
        
        return np.array(force_data)
    
    @cached_property
    def stress(self) -> np.ndarray | None:
        """Parse stress tensor from vasprun.xml
        
        Returns array with shape (3, 3)
        """
        tag_list = self.root.xpath("//calculation/stress/varray[@name='stress']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Parse stress tensor
        stress_data: list[list[float]] = []
        for v_elem in element:
            if v_elem.tag == "v" and v_elem.text:
                values = v_elem.text.split()
                stress_data.append([float(v) for v in values])
        
        return np.array(stress_data)
    
    @cached_property
    def e_fr_energy(self) -> float | None:
        """Parse free energy from vasprun.xml"""
        tag_list = self.root.xpath("//calculation/energy/i[@name='e_fr_energy']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        if element.text:
            return float(element.text.strip())
        return None
    
    @cached_property
    def e_wo_entrp(self) -> float | None:
        """Parse energy without entropy from vasprun.xml"""
        tag_list = self.root.xpath("//calculation/energy/i[@name='e_wo_entrp']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        if element.text:
            return float(element.text.strip())
        return None
    
    @cached_property
    def e_0_energy(self) -> float | None:
        """Parse energy at 0K from vasprun.xml"""
        tag_list = self.root.xpath("//calculation/energy/i[@name='e_0_energy']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        if element.text:
            return float(element.text.strip())
        return None
    
    @cached_property
    def initial_structure(self) -> StructureInfo | None:
        """Parse initial structure from vasprun.xml"""
        tag_list = self.root.xpath("//calculation/structure")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        # Parse crystal info
        basis_data: list[list[float]] = []
        rec_basis_data: list[list[float]] = []
        volume = 0.0
        positions_data: list[list[float]] = []
        
        for child in element:
            if child.tag == "crystal":
                for subchild in child:
                    if subchild.tag == "varray":
                        name = subchild.attrib.get("name", "")
                        if name == "basis":
                            for v_elem in subchild:
                                if v_elem.tag == "v" and v_elem.text:
                                    basis_data.append([float(x) for x in v_elem.text.split()])
                        elif name == "rec_basis":
                            for v_elem in subchild:
                                if v_elem.tag == "v" and v_elem.text:
                                    rec_basis_data.append([float(x) for x in v_elem.text.split()])
                    elif subchild.tag == "i":
                        if subchild.attrib.get("name") == "volume" and subchild.text:
                            volume = float(subchild.text.strip())
            elif child.tag == "varray":
                if child.attrib.get("name") == "positions":
                    for v_elem in child:
                        if v_elem.tag == "v" and v_elem.text:
                            positions_data.append([float(x) for x in v_elem.text.split()])
        
        crystal = CrystalInfo(
            basis=np.array(basis_data),
            volume=volume,
            rec_basis=np.array(rec_basis_data)
        )
        
        return StructureInfo(
            crystal=crystal,
            positions=np.array(positions_data)
        )
    
    @cached_property
    def structure_element(self) -> dict[str, Any] | None:
        """Parse structure element from vasprun.xml"""
        tag_list = self.root.xpath("//structure[@name='initialpos']")
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return None
        
        element = tag_list[0]
        assert isinstance(element, etree._Element)
        
        name = element.attrib.get("name", "")
        
        # Parse crystal info
        basis_data: list[list[float]] = []
        rec_basis_data: list[list[float]] = []
        volume = 0.0
        positions_data: list[list[float]] = []
        
        for child in element:
            if child.tag == "crystal":
                for subchild in child:
                    if subchild.tag == "varray":
                        varray_name = subchild.attrib.get("name", "")
                        if varray_name == "basis":
                            for v_elem in subchild:
                                if v_elem.tag == "v" and v_elem.text:
                                    basis_data.append([float(x) for x in v_elem.text.split()])
                        elif varray_name == "rec_basis":
                            for v_elem in subchild:
                                if v_elem.tag == "v" and v_elem.text:
                                    rec_basis_data.append([float(x) for x in v_elem.text.split()])
                    elif subchild.tag == "i":
                        if subchild.attrib.get("name") == "volume" and subchild.text:
                            volume = float(subchild.text.strip())
            elif child.tag == "varray":
                if child.attrib.get("name") == "positions":
                    for v_elem in child:
                        if v_elem.tag == "v" and v_elem.text:
                            positions_data.append([float(x) for x in v_elem.text.split()])
        
        return {
            "name": name,
            "crystal": {
                "basis": basis_data,
                "volume": volume,
                "rec_basis": rec_basis_data
            },
            "positions": positions_data
        }
    
    @cached_property
    def self_consistent_steps(self) -> list[SCStepInfo]:
        """Parse self-consistent steps from vasprun.xml"""
        tag_list = self.root.xpath("//calculation/scstep")
        assert isinstance(tag_list, Iterable)
        
        steps: list[SCStepInfo] = []
        for element in tag_list:
            if not isinstance(element, etree._Element):
                continue
            
            # Parse time info
            dav_time = 0.0
            total_time = 0.0
            
            # Parse energy info
            energy_dict: dict[str, float] = {}
            
            for child in element:
                if child.tag == "time":
                    name = child.attrib.get("name", "")
                    if child.text:
                        values = child.text.split()
                        if values:
                            if name == "dav":
                                dav_time = float(values[0])
                            elif name == "total":
                                total_time = float(values[0])
                elif child.tag == "energy":
                    for energy_elem in child:
                        if energy_elem.tag == "i":
                            ename = energy_elem.attrib.get("name", "")
                            if energy_elem.text:
                                energy_dict[ename] = float(energy_elem.text.strip())
            
            time_info = TimeInfo(dav=dav_time, total=total_time)
            energy_info = EnergyInfo(
                e_fr_energy=energy_dict.get("e_fr_energy", energy_dict.get("alphaZ", 0.0)),
                e_wo_entrp=energy_dict.get("e_wo_entrp", energy_dict.get("alphaZ", 0.0)),
                e_0_energy=energy_dict.get("e_0_energy", energy_dict.get("alphaZ", 0.0)),
                alphaZ=energy_dict.get("alphaZ"),
                ewald=energy_dict.get("ewald"),
                hartreedc=energy_dict.get("hartreedc"),
                XCdc=energy_dict.get("XCdc"),
                pawpsdc=energy_dict.get("pawpsdc"),
                pawaedc=energy_dict.get("pawaedc"),
                eentropy=energy_dict.get("eentropy"),
                bandstr=energy_dict.get("bandstr"),
                atom=energy_dict.get("atom")
            )
            
            steps.append(SCStepInfo(time=time_info, energy=energy_info))
        
        return steps

#     @property
#     def is_spin_polarized(self):
#         spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
#         return len(spins) == 2
    

#     @property
#     def bands(self) -> dict[str, dict[str, np.ndarray]]:
#         """Parses the electronic bands

#         Returns
#         -------
#         np.ndarray
#             The electronic bands
#         """
#         spins = list(self.data["general"]["eigenvalues"]["array"]["data"].keys())
#         kpoints_list = list(
#             self.data["general"]["eigenvalues"]["array"]["data"]["spin 1"].keys()
#         )
#         eigen_values: dict[str, dict[str, np.ndarray]] = {}
#         nbands = len(
#             self.data["general"]["eigenvalues"]["array"]["data"][spins[0]][
#                 kpoints_list[0]
#             ][kpoints_list[0]]
#         )
#         nkpoints = len(kpoints_list)
#         for ispin in spins:
#             eigen_values[ispin] = {}
#             eigen_values[ispin]["eigen_values"] = np.zeros(shape=(nbands, nkpoints))
#             eigen_values[ispin]["occupancies"] = np.zeros(shape=(nbands, nkpoints))
#             for ikpoint, kpt in enumerate(kpoints_list):
#                 temp = np.array(
#                     self.data["general"]["eigenvalues"]["array"]["data"][ispin][kpt][
#                         kpt
#                     ]
#                 )
#                 if self.fermi is not None:
#                     eigen_values[ispin]["eigen_values"][:, ikpoint] = (
#                         temp[:, 0] - self.fermi
#                     )
#                 else:
#                     eigen_values[ispin]["eigen_values"][:, ikpoint] = temp[:, 0]
#                 eigen_values[ispin]["occupancies"][:, ikpoint] = temp[:, 1]
#         return eigen_values

#     @property
#     def bands_projected(self) -> dict[str, np.ndarray]:
#         """Parse the band projections

#         Returns
#         -------
#         np.ndarray
#             The band projections
#         """
#         # projected[iatom][ikpoint][iband][iprincipal][iorbital][ispin]
#         labels = self.data["general"]["projected"]["array"]["info"]
#         spins = list(self.data["general"]["projected"]["array"]["data"].keys())
#         kpoints_list = list(
#             self.data["general"]["projected"]["array"]["data"][spins[0]].keys()
#         )
#         bands_list = list(
#             self.data["general"]["projected"]["array"]["data"][spins[0]][
#                 kpoints_list[0]
#             ][kpoints_list[0]].keys()
#         )
#         bands_projected = {"labels": labels}

#         nspins = len(spins)
#         nkpoints = len(kpoints_list)
#         nbands = len(bands_list)
#         norbitals = len(labels)
#         natoms: int = self.initial_structure.natoms
#         bands_projected["projection"] = np.zeros(
#             shape=(nspins, nkpoints, nbands, natoms, norbitals)
#         )
#         for ispin, spn in enumerate(spins):
#             for ikpoint, kpt in enumerate(kpoints_list):
#                 for iband, bnd in enumerate(bands_list):
#                     bands_projected["projection"][ispin, ikpoint, iband, :, :] = (
#                         np.array(
#                             self.data["general"]["projected"]["array"]["data"][spn][
#                                 kpt
#                             ][kpt][bnd][bnd]
#                         )
#                     )
#         # ispin, ikpoint, iband, iatom, iorbital
#         bands_projected["projection"] = np.swapaxes(bands_projected["projection"], 0, 3)
#         # iatom, ikpoint, iband, ispin, iorbital
#         bands_projected["projection"] = np.swapaxes(bands_projected["projection"], 3, 4)
#         # iatom, ikpoint, iband, iorbital, ispin
#         bands_projected["projection"] = bands_projected["projection"].reshape(
#             natoms, nkpoints, nbands, 1, norbitals, nspins
#         )

#         return bands_projected

#     def _get_dos_total(self):
#         """A helper method to get the total density of states

#         Returns
#         -------
#         tuple
#             Returns the dos_total info as a dict and the a list of labels
#         """
#         spins = list(self.data["general"]["dos"]["total"]["array"]["data"].keys())
#         energies = np.array(
#             self.data["general"]["dos"]["total"]["array"]["data"][spins[0]]
#         )[:, 0]
#         dos_total = {"energies": energies}

#         for spin_name in spins:
#             dos_total[self.spins_dict[spin_name]] = np.array(
#                 self.data["general"]["dos"]["total"]["array"]["data"][spin_name]
#             )[:, 1]

#         return dos_total, list(dos_total.keys())

#     def _get_dos_projected(self, 
#                            atoms: list[int] | None = None
#                            ) -> tuple[dict[str, Any] | None, list[str] | None]:
#         """A helper method to get the projected density of states

#         Parameters
#         ----------
#         atoms : list, optional
#             List of atoms, by default []

#         Returns
#         -------
#         _type_
#             Returns the dos_total info as a dict and the a list of labels
#         """
#         assert atoms is not None, "atoms is required"
#         if len(atoms) == 0:
#             atoms = list(np.arange(self.initial_structure.natoms).astype(int))

#         if "partial" in self.data["general"]["dos"]:
#             dos_projected: dict[str, Any] = {}
#             ion_list = [
#                 f"ion {x + 1}" for x in atoms
#             ]  # using this name as vasrun.xml uses ion #
#             for i in range(len(ion_list)):
#                 iatom = ion_list[i]
#                 name = self.initial_structure.atoms[atoms[i]] + str(atoms[i])
#                 spins = list(
#                     self.data["general"]["dos"]["partial"]["array"]["data"][
#                         iatom
#                     ].keys()
#                 )
#                 energies = np.array(
#                     self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
#                         spins[0]
#                     ][spins[0]]
#                 )[:, 0]
#                 dos_projected[name] = {"energies": energies}
#                 for ispin in spins:
#                     dos_projected[name][self.spins_dict[ispin]] = np.array(
#                         self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
#                             ispin
#                         ][ispin]
#                     )[:, 1:]
                    
   
#                 # if 'Spin-Total' in list(dos_projected[name].keys()):
#                 #     del dos_projected[name]['Spin-Total']
#             return (
#                 dos_projected,
#                 self.data["general"]["dos"]["partial"]["array"]["info"],
#             )
#         else:
#             print("This calculation does not include partial density of states")
#             return None, None

#     @cached_property
#     def total_dos(self) -> np.ndarray:
#         total: list[np.ndarray] = []
#         for ispin in self.dos_total:
#             if ispin == "energies":
#                 continue
#             total.append(self.dos_total[ispin])
#         return np.array(total)

#     @property
#     def dos_to_dict(self) -> dict[str, Any]:
#         """
#         The complete density (total,projected) of states as a python dictionary

#         Returns
#         -------
#         dict
#              The complete density (total,projected) of states as a python dictionary
#         """

#         return {"total": self._get_dos_total(), "projected": self._get_dos_projected(atoms=[])}

#     @property
#     def dos_total(self) -> dict[str, Any]:
#         """Returns the total dos dict

#         Returns
#         -------
#         dict
#             Returns the total dos dict
#         """
#         dos_total, _ = self._get_dos_total()
#         return dos_total

#     @property
#     def dos_projected(self) -> np.ndarray | None:
#         """
#         Returns the projected DOS as a multi-dimentional array, to be used in the
#         pyprocar.core.dos object

#         Returns
#         -------
#         np.ndarray
#             Returns the projected DOS as a multi-dimentional array, to be used in the
#             pyprocar.core.dos object
#         """
#         ret: list[list[list[np.ndarray]]] = []
#         dos_projected, info = self._get_dos_projected(atoms=[])
#         if dos_projected is None:
#             return None
#         assert info is not None, "info is required"
#         norbitals = len(info) - 1
#         info[0] = info[0].capitalize()
#         labels: list[str] = []
#         labels.append(info[0])
#         for iatom in dos_projected:
#             temp_atom: list[list[np.ndarray]] = []
#             for iorbital in range(norbitals):
#                 temp_spin: list[np.ndarray] = []
#                 for key in dos_projected[iatom]:
#                     if key == "energies":
#                         continue
#                     temp_spin.append(dos_projected[iatom][key][:, iorbital])
#                 temp_atom.append(temp_spin)
#             ret.append(temp_atom)
#         return np.array(ret)

#     @property
#     def kpoints(self) -> dict[str, Any]:
#         """
#         Returns the kpoints used in the calculation in form of a pychemia.core.KPoints object

#         Returns
#         -------
#         np.ndarray
#             Returns the kpoints used in the calculation
#             in form of a pychemia.core.KPoints object
#         """

#         if self.data["kpoints_info"]["mode"] == "listgenerated":
#             kpoints = dict(
#                 mode="path", kvertices=self.data["kpoints_info"]["kpoint_vertices"]
#             )
#         else:
#             kpoints = dict(
#                 mode=self.data["kpoints_info"]["mode"].lower(),
#                 grid=self.data["kpoints_info"]["kgrid"],
#                 shifts=self.data["kpoints_info"]["user_shift"],
#             )
#         return kpoints

#     @property
#     def kpoints_list(self) -> dict[str, Any]:
#         """
#         Returns the dict of kpoints and weights used in the calculation
#         in form of a pychemia.core.KPoints object

#         Returns
#         -------
#         dict
#             Returns a dict of kpoints information
#             in form of a pychemia.core.KPoints object
#         """
#         return dict(
#             mode="reduced",
#             kpoints_list=self.data["kpoints"]["kpoints_list"],
#             weights=self.data["kpoints"]["k_weights"],
#         )

#     @property
#     def incar(self) -> dict[str, Any]:
#         """
#         Returns the incar parameters used in the calculation as pychemia.code.vasp.VaspIncar object

#         Returns
#         -------
#         Description
#             Returns the incar parameters
#         """
#         return self.data["incar"]

#     @property
#     def vasp_parameters(self) -> dict[str, Any]:
#         """
#         Returns all of the parameters vasp has used in this calculation
#         """
#         return self.data["vasp_params"]

#     @property
#     def potcar_info(self) -> list[str]:
#         """
#         Returns the information about pseudopotentials(POTCAR) used in this calculation
#         """
#         return self.data["atom_info"]["atom_types"]

#     @property
#     def fermi(self) -> float | None:
#         """
#         Returns the fermi energy
#         """
        
#         if "fermi" in self.data["general"]["dos"]:
#             return self.data["general"]["dos"]["fermi"]
#         elif "efermi" in self.data["general"]["dos"]:
#             return self.data["general"]["dos"]["efermi"]
#         else:
#             return None

#     @property
#     def species(self) -> list[str]:
#         """
#         Returns the species in POSCAR
#         """
#         return list(self.initial_structure.species)

#     @property
#     def structures(self) -> list[Structure]:
#         """
#         Returns a list of pychemia.core.Structure representing all the ionic step structures
#         """
#         symbols = [x.strip() for x in self.data["atom_info"]["symbols"]]
#         structures: list[Structure] = []
#         for ist in self.data["structures"]:

#             st = Structure(
#                 atoms=symbols,
#                 fractional_coordinates=ist["reduced"],
#                 lattice=ist["cell"],
#             )
#             structures.append(st)
#         return structures

#     @property
#     def structure(self) -> Structure:
#         """
#         crystal structure of the last step
#         """
#         return self.structures[-1]

#     @property
#     def forces(self) -> list[np.ndarray]:
#         """
#         Returns all the forces in ionic steps
#         """
#         return self.data["forces"]

#     @property
#     def initial_structure(self) -> Structure:
#         """
#         Returns the initial Structure as a pychemia structure
#         """
#         return self.structures[0]

#     @property
#     def final_structure(self) -> Structure:
#         """
#         Returns the final Structure as a pychemia structure
#         """

#         return self.structures[-1]

#     @property
#     def iteration_data(self) -> list[dict[str, Any]]:
#         """
#         Returns a list of information in each electronic and ionic step of calculation
#         """
#         return self.data["calculation"]

#     @property
#     def energies(self) -> list[list[float]]:
#         """
#         Returns a list of energies in each electronic and ionic step 
#         [ionic step,electronic step, energy]
#         """
#         scf_step = 0
#         ion_step = 0
#         double_counter = 1
#         energies: list[list[float]] = []
#         for calc in self.data["calculation"]:
#             if "ewald" in calc["energy"]:
#                 if double_counter == 0:
#                     double_counter += 1
#                     scf_step += 1
#                 elif double_counter == 1:
#                     double_counter = 0
#                     ion_step += 1
#                     scf_step = 1
#             else:
#                 scf_step += 1
#             energies.append([ion_step, scf_step, calc["energy"]["e_0_energy"]])
#         return energies

#     @property
#     def last_energy(self):
#         """
#         Returns the last calculated energy of the system
#         """
#         return self.energies[-1][-1]

#     @property
#     def energy(self):
#         """
#         Returns the last calculated energy of the system
#         """
#         return self.last_energy

#     @property
#     def convergence_electronic(self):
#         """
#         Returns a boolian representing if the last electronic self-consistent
#         calculation converged
#         """
#         ediff = self.vasp_parameters["electronic"]["EDIFF"]
#         last_dE = abs(self.energies[-1][-1] - self.energies[-2][-1])
#         return last_dE < ediff

#     @property
#     def convergence_ionic(self) -> bool:
#         """
#         Returns a boolian representing if the ionic part of the
#         calculation converged
#         """
#         energies = np.array(self.energies)
#         nsteps = len(np.unique(np.array(self.energies)[:, 0]))
#         if nsteps == 1:
#             print("This calculation does not have ionic steps")
#             return True
#         else:
#             ediffg = self.vasp_parameters["ionic"]["EDIFFG"]
#             if ediffg < 0:
#                 last_forces_abs = np.abs(np.array(self.forces[-1]))
#                 return not (np.any(last_forces_abs > abs(ediffg)))
#             else:
#                 last_ionic_energy = energies[(energies[:, 0] == nsteps)][-1][-1]
#                 penultimate_ionic_energy = energies[(energies[:, 0] == (nsteps - 1))][
#                     -1
#                 ][-1]
#                 last_dE = abs(last_ionic_energy - penultimate_ionic_energy)
#                 if last_dE < ediffg:
#                     return True
#         return False

#     @property
#     def convergence(self) -> bool:
#         """
#         Returns a boolian representing if the the electronic self-consistent
#         and ionic calculation converged
#         """
#         return self.convergence_electronic and self.convergence_ionic

#     @property
#     def is_finished(self) -> bool:
#         """
#         Always returns True, need to fix this according to reading the xml as if the calc is
#         not finished we will have errors in xml parser
#         """
#         # if vasprun.xml is read the calculation is finished
#         return True
    
#     @cached_property
#     def calculation(self) -> list[dict[str, Any]]:
#         return self.data["calculation"]

#     @cached_property
#     def run_info(self) -> dict[str, Any]:
#         return self.data["run_info"]

#     @cached_property
#     def general(self) -> dict[str, Any]:
#         return self.data["general"]
    
#     @cached_property
#     def kpoints_info(self) -> dict[str, Any]:
#         return self.data["kpoints_info"]
    
#     @cached_property
#     def vasp_params(self) -> dict[str, Any]:
#         return self.data["vasp_params"]
    
#     @cached_property
#     def atom_info(self) -> dict[str, Any]:
#         return self.data["atom_info"]

#     def text_to_bool(self, text: str) -> bool:
#         """boolians in vaspxml are stores as T or F in str format, 
#         this function coverts them to python boolians"""
#         text = text.strip(" ")
#         return text == "T" or text == ".True." or text == ".TRUE."

#     def conv(self, ele: str, _type: str) -> float | int | str | bool | None:
#         """This function converts the xml text to the type specified in the attrib of xml tree"""

#         if _type == "string":
#             return ele.strip()
#         elif _type == "int":
#             return int(ele)
#         elif _type == "logical":
#             return self.text_to_bool(ele)
#         elif _type == "float":
#             if "*" in ele:
#                 return float('nan')
#             else:
#                 return float(ele)

#     def get_varray(self, element: etree.Element) -> list[list[float]]:
#         """Returns an array for each varray tag in vaspxml"""
#         ret: list[list[float]] = []
#         for subelement in element:
#             tmp: list[float] = []
#             assert subelement.text is not None, "subelement.text is required"
#             for x in subelement.text.split():
#                 try:
#                     tmp.append(float(x))
#                 except ValueError:
#                     tmp.append(0.0)
#             ret.append(tmp)
#         return ret

#     def get_params(self, element: etree.Element, dest: dict[str, Any]) -> dict[str, Any]:
#         """dest should be a dictionary
#         This function is recurcive #check spelling"""
#         for subelement in element:
#             if subelement.tag == "separator":
#                 dest[subelement.attrib["name"].strip()] = {}
#                 dest[subelement.attrib["name"].strip()] = self.get_params(
#                     subelement, dest[subelement.attrib["name"]]
#                 )
#             else:
#                 _type  = subelement.attrib.get("type", "float")
#                 if subelement.text is None:
#                     dest[subelement.attrib["name"].strip()] = None

#                 elif len(subelement.text.split()) > 1:
#                     dest[subelement.attrib["name"].strip()] = [
#                         self.conv(x, _type) for x in subelement.text.split()
#                     ]
#                 else:
#                     dest[subelement.attrib["name"].strip()] = self.conv(
#                         subelement.text, _type
#                     )

#         return dest

#     def get_structure(self, element: etree.Element) -> dict[str, Any]:
#         """Returns a dictionary of the structure"""
#         ret: dict[str, Any] = {}
#         for subelement in element:
#             if subelement.tag == "crystal":
#                 for subsubelement in subelement:
#                     if subsubelement.attrib["name"] == "basis":
#                         ret["cell"] = self.get_varray(subsubelement)
#                     elif subsubelement.attrib["name"] == "volume":
#                         assert subsubelement.text is not None, "subsubelement.text is required"
#                         ret["volume"] = float(subsubelement.text)
#                     elif subsubelement.attrib["name"] == "rec_basis":
#                         ret["rec_cell"] = self.get_varray(subsubelement)
#             elif subelement.tag == "varray" and subelement.attrib["name"] == "positions":
#                 ret["reduced"] = self.get_varray(subelement)
#         return ret

#     def get_scstep(self, element: etree.Element) -> dict[str, Any]:
#         """This function extracts the self-consistent step information"""
#         scstep: dict[str, Any] = {"time": {}, "energy": {}}
#         for subelement in element:
#             if subelement.tag == "time":
#                 assert subelement.text is not None, "subelement.text is required"
#                 scstep["time"][subelement.attrib["name"]] = [
#                     float(x) for x in subelement.text.split()
#                 ]
#             elif subelement.tag == "energy":
#                 for subsubelement in subelement:
#                     assert subsubelement.text is not None, "subsubelement.text is required"
#                     scstep["energy"][subsubelement.attrib["name"]] = float(subsubelement.text)
#         return scstep

#     def get_set(self, element: etree.Element, ret: dict[str, Any]) -> dict[str, Any]:
#         """This function will extract any element taged set recurcively"""
#         if len(element) == 0:
#             return ret
#         if element[0].tag == "r":
#             ret[element.attrib["comment"]] = self.get_varray(element)
#             return ret
#         else:
#             ret[element.attrib["comment"]] = {}
#             for subelement in element:

#                 if subelement.tag == "set":
#                     ret[element.attrib["comment"]][subelement.attrib["comment"]] = {}
#                     ret[element.attrib["comment"]][subelement.attrib["comment"]] = (
#                         self.get_set(
#                             subelement,
#                             ret[element.attrib["comment"]][subelement.attrib["comment"]],
#                         )
#                     )
#             return ret

#     def get_general(self, element: etree.Element, ret: dict[str, Any]) -> dict[str, Any]:
#         """This function will parse any element in calculatio other than the structures, scsteps"""
#         # Check if this is an array with field or dimension tags
#         subelement_tags = [subelement.tag for subelement in element]
#         if "dimension" in subelement_tags or "field" in subelement_tags:
#             info_list: list[str] = []
#             ret["data"] = {}
#             for subelement in element:
#                 if subelement.tag == "field":
#                     assert subelement.text is not None, "subelement.text is required"
#                     info_list.append(subelement.text.strip(" "))
#                 elif subelement.tag == "dimension":
#                     # Handle dimension tags if needed
#                     pass
#                 elif subelement.tag == "set":
#                     for subsubelement in subelement:
#                         ret["data"] = self.get_set(subsubelement, ret["data"])
#             ret["info"] = info_list
#             return ret
#         else:
#             for subelement in element:
#                 if subelement.tag == "i":
#                     if "name" in subelement.attrib and subelement.attrib["name"] == "fermi":
#                         assert subelement.text is not None, "subelement.text is required"
#                         ret["fermi"] = float(subelement.text)
#                     continue
#                 if subelement.tag not in ret:
#                     ret[subelement.tag] = {}
#                 ret[subelement.tag] = self.get_general(subelement, ret[subelement.tag])
#             return ret

#     def _parse_vasprun(self) -> dict[str, Any]:
#         if self.filepath is not None:
#             tree = etree.parse(str(self.filepath))
#             root = tree.getroot()
#         else:
#             root = etree.fromstring(self.file_str.encode('utf-8'))

#         calculation: list[dict[str, Any]] = []
#         structures: list[dict[str, Any]] = []
#         forces: list[list[list[float]]] = []
#         stresses: list[list[list[float]]] = []
#         orbital_magnetization: dict[str, Any] = {}
#         run_info: dict[str, Any] = {}
#         incar: dict[str, Any] = {}
#         general: dict[str, Any] = {}
#         kpoints_info: dict[str, Any] = {}
#         vasp_params: dict[str, Any] = {}
#         kpoints_list: np.ndarray | list[list[float]] = []
#         k_weights: np.ndarray | list[float] = []
#         atom_info: dict[str, Any] = {}
#         for element in root:

#             if element.tag == "generator":
#                 for subelement in element:
#                     run_info[subelement.attrib["name"]] = subelement.text

#             elif element.tag == "incar":
#                 incar = self.get_params(element, incar)

#             # Skipping 1st structure which is primitive cell
#             elif element.tag == "kpoints":
#                 for subelement in element:
#                     tag_name = subelement.tag
#                     subelement_items = list(subelement.items())
#                     first_item_key = None
#                     first_item_value = None
#                     if len(subelement_items) > 0:
#                         first_item_key = subelement_items[0][0]
#                         first_item_value = subelement_items[0][1]
                        
#                     if tag_name == "generation" or (first_item_key is not None 
#                                                     and first_item_key == "param"):
#                         kpoints_info["mode"] = first_item_value
#                         if kpoints_info["mode"] == "listgenerated":
#                             kpoint_vertices_list: list[list[float]] = []
#                             for subsubelement in subelement:

#                                 if ("name" in subsubelement.attrib 
#                                     and subsubelement.attrib["name"] == "divisions"):
#                                     assert subsubelement.text is not None
#                                     kpoints_info["ndivision"] = int(subsubelement.text)
#                                 else:
#                                     assert subsubelement.text is not None
#                                     if len(subsubelement.text.split()) != 3:
#                                         continue
#                                     kpoint_vertices_list.append(
#                                         [float(x) for x in subsubelement.text.split()]
#                                     )
#                             kpoints_info["kpoint_vertices"] = kpoint_vertices_list
#                         else:
#                             for subsubelement in subelement:
#                                 if subsubelement.attrib["name"] == "divisions":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["kgrid"] = [
#                                         int(x) for x in subsubelement.text.split()
#                                     ]
#                                 elif subsubelement.attrib["name"] == "usershift":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["user_shift"] = [
#                                         float(x) for x in subsubelement.text.split()
#                                     ]
#                                 elif subsubelement.attrib["name"] == "genvec1":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["genvec1"] = [
#                                         float(x) for x in subsubelement.text.split()
#                                     ]
#                                 elif subsubelement.attrib["name"] == "genvec2":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["genvec2"] = [
#                                         float(x) for x in subsubelement.text.split()
#                                     ]
#                                 elif subsubelement.attrib["name"] == "genvec3":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["genvec3"] = [
#                                         float(x) for x in subsubelement.text.split()
#                                     ]
#                                 elif subsubelement.attrib["name"] == "shift":
#                                     assert subsubelement.text is not None
#                                     kpoints_info["shift"] = [
#                                         float(x) for x in subsubelement.text.split()
#                                     ]
         

#                     elif (tag_name == "varray" and first_item_key is not None 
#                           and first_item_key == "kpointlist"):
#                         temp_kpoints_list: list[list[float]] = []
#                         for subsubelement in subelement:
#                             assert subsubelement.text is not None, "subsubelement.text is required"
#                             temp_kpoints_list.append([float(x) for x in subsubelement.text.split()])
#                         kpoints_list = np.array(temp_kpoints_list)
#                     elif (tag_name == "varray" and first_item_key is not None 
#                           and first_item_key == "weights"):
#                         temp_k_weights: list[float] = []
#                         for subsubelement in subelement:
#                             assert subsubelement.text is not None, "subsubelement.text is required"
#                             temp_k_weights.append(float(subsubelement.text))
#                         k_weights = np.array(temp_k_weights)
                        
#                     elif tag_name == "kpoints_labels":
#                         kpoint_labels_list: list[str] = []
#                         for subsubelement in subelement:
#                             subsubelement_items = list(subsubelement.items())
#                             kpoint_label = None
#                             if len(subsubelement_items) > 0:
#                                 kpoint_label = subsubelement_items[0][1]
#                             if kpoint_label is not None:
#                                 kpoint_labels_list.append(kpoint_label)
#                         kpoints_info["kpoint_labels"] = kpoint_labels_list

#             # Vasp Parameters
#             elif element.tag == "parameters":
#                 vasp_params = self.get_params(element, vasp_params)

#             # Atom info
#             elif element.tag == "atominfo":

#                 for subelement in element:
#                     if subelement.tag == "atoms":
#                         assert subelement.text is not None
#                         atom_info["natom"] = int(subelement.text)
#                     elif subelement.tag == "types":
#                         assert subelement.text is not None
#                         atom_info["nspecies"] = int(subelement.text)
#                     elif subelement.tag == "array":
#                         if subelement.attrib["name"] == "atoms":
#                             for subsubelement in subelement:
#                                 if subsubelement.tag == "set":
#                                     symbols_list: list[str] = []
#                                     for subsubsubelement in subsubelement:
#                                         assert subsubsubelement[0].text is not None
#                                         symbols_list.append(subsubsubelement[0].text)
#                                     atom_info["symbols"] = symbols_list
#                         elif subelement.attrib["name"] == "atomtypes":
#                             atom_info["atom_types"] = {}
#                             for subsubelement in subelement:
#                                 if subsubelement.tag == "set":
#                                     for subsubsubelement in subsubelement:
#                                         atom_info["atom_types"][subsubsubelement[1].text] = {}
#                                         assert subsubsubelement[0].text is not None
#                                         atom_info["atom_types"][subsubsubelement[1].text][
#                                             "natom_per_specie"
#                                         ] = int(subsubsubelement[0].text)
#                                         assert subsubsubelement[2].text is not None
#                                         atom_info["atom_types"][subsubsubelement[1].text][
#                                             "mass"
#                                         ] = float(subsubsubelement[2].text)
#                                         assert subsubsubelement[3].text is not None
#                                         atom_info["atom_types"][subsubsubelement[1].text][
#                                             "valance"
#                                         ] = float(subsubsubelement[3].text)
#                                         assert subsubsubelement[4].text is not None
#                                         atom_info["atom_types"][subsubsubelement[1].text][
#                                             "pseudopotential"
#                                         ] = subsubsubelement[4].text.strip()

#             elif element.tag == "structure":
#                 # if element.attrib["name"] == "initialpos":
#                 #     initial_pos = self.get_structure(element)
#                 # elif element.attrib["name"] == "finalpos":
#                 #     final_pos = self.get_structure(element)
#                 pass

#             elif element.tag == "calculation":
#                 for subelement in element:
#                     if subelement.tag == "scstep":
#                         calculation.append(self.get_scstep(subelement))
#                     elif subelement.tag == "structure":
#                         structures.append(self.get_structure(subelement))
#                     elif subelement.tag == "varray":
#                         if subelement.attrib["name"] == "forces":
#                             forces.append(self.get_varray(subelement))
#                         elif subelement.attrib["name"] == "stress":
#                             stresses.append(self.get_varray(subelement))

#                     # elif ielement.tag == 'eigenvalues':
#                     #     for isub in ielement[0] :
#                     #         if isub.tag == 'set':
#                     #             for iset in isub :
#                     #                 eigen_values[iset.attrib['comment']] = {}
#                     #                 for ikpt in iset :
#                     #                     eigen_values[iset.attrib['comment']]
#                     # [ikpt.attrib['comment']] = get_varray(ikpt)

#                     elif subelement.tag == "separator":
#                         if subelement.attrib["name"] == "orbital magnetization":
#                             for subsubelement in subelement:
#                                 assert subsubelement.text is not None
#                                 orbital_magnetization[subsubelement.attrib["name"]] = [
#                                     float(x) for x in subsubelement.text.split()
#                                 ]

#                     # elif ielement.tag == 'dos':
#                     #     for isub in ielement :
#                     #         if 'name' in isub.attrib:
#                     #             if isub.attrib['name'] == 'fermi' :
#                     #                 dos['fermi'] = float(isub.text)
#                     #             else :
#                     #                 dos[isub.tag] = {}
#                     #                 dos[isub.tag]['info'] = []
#                     #               for iset in isub[0]  :
#                     #                   if iset.tag == 'set' :
#                     #                       for isub_set in iset:
#                     #                           dos[isub.tag] = get_set(isub_set,dos[isub.tag])
#                     #                   elif iset.tag == 'field' :
#                     #                       dos[isub.tag]['info'].append(iset.text.strip(' '))
#                     else:
#                         general[subelement.tag] = {}
#                         general[subelement.tag] = self.get_general(
#                             subelement, general[subelement.tag]
#                         )
#             # NEED TO ADD ORBITAL MAGNETIZATION

#         return {
#             "calculation": calculation,
#             "structures": structures,
#             "forces": forces,
#             "run_info": run_info,
#             "incar": incar,
#             "general": general,
#             "kpoints_info": kpoints_info,
#             "vasp_params": vasp_params,
#             "kpoints": {"kpoints_list": kpoints_list, "k_weights": k_weights},
#             "atom_info": atom_info,
#         }

    @override
    def __contains__(self, key: object) -> bool:
        return key in self.__dict__

    @override
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    @override
    def __len__(self) -> int:
        return len(self.__dict__)