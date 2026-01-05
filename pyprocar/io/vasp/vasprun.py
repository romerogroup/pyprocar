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
        xml_str = re.sub(r"<\?xml[^>]+\?>", "", self._file_str)
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

            if name not in [
                "GGA_COMPAT",
                "LBERRY",
                "ICORELEVEL",
                "LDAU",
                "LDAUTYPE",
                "LDAUL",
                "LDAUU",
                "LDAUJ",
                "LDAUPRINT",
                "I_CONSTRAINED_M",
            ]:
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
        tag_list = self.root.xpath(
            "//parameters/separator" + "[@name='electronic exchange-correlation']"
        )
        assert isinstance(tag_list, Iterable)
        if not tag_list:
            return {}
        element = tag_list[-1]
        assert isinstance(element, etree._Element)
        params: dict[str, Any] = {}
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
            time=info.get("time", ""),
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
                    if subsubelement.attrib.get("name") == "divisions" and subsubelement.text:
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
            kshift=kshift,
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
    def dos_energies(self) -> np.ndarray | None:
        """Parse DOS energy grid from vasprun.xml

        Returns array with shape (n_energies,)
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

        # Get energies from first spin set
        for spin_set in set_elem:
            if spin_set.tag == "set":
                energy_data: list[float] = []
                for r_elem in spin_set:
                    if r_elem.tag == "r" and r_elem.text:
                        values = r_elem.text.split()
                        energy_data.append(float(values[0]))  # energy is first value
                return np.array(energy_data)

        return None

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
            elif child.tag == "varray" and child.attrib.get("name") == "positions":
                for v_elem in child:
                    if v_elem.tag == "v" and v_elem.text:
                        positions_data.append([float(x) for x in v_elem.text.split()])

        crystal = CrystalInfo(
            basis=np.array(basis_data), volume=volume, rec_basis=np.array(rec_basis_data)
        )

        return StructureInfo(crystal=crystal, positions=np.array(positions_data))

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
            elif child.tag == "varray" and child.attrib.get("name") == "positions":
                for v_elem in child:
                    if v_elem.tag == "v" and v_elem.text:
                        positions_data.append([float(x) for x in v_elem.text.split()])

        return {
            "name": name,
            "crystal": {"basis": basis_data, "volume": volume, "rec_basis": rec_basis_data},
            "positions": positions_data,
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
                atom=energy_dict.get("atom"),
            )

            steps.append(SCStepInfo(time=time_info, energy=energy_info))

        return steps

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
