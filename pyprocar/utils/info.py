# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List, Union

orbital_names = {'s': [0],
                 'p': [1, 2, 3],
                 'd': [4, 5, 6, 7, 8],
                 'f': [9, 10, 11, 12, 13, 14, 15],
                 "s": 0,
                 "py": 1,
                 "pz": 2,
                 "px": 3,
                 "dxy": 4,
                 "dyz": 5,
                 "dz2": 6,
                 "dxz": 7,
                 "x2-y2": 8,
                 "fy3x2": 9,
                 "fxyz": 10,
                 "fyz2": 11,
                 "fz3": 12,
                 "fxz2": 13,
                 "fzx2": 14,
                 "fxx": 15,
                 }



AZIMUTHAL_ORBITAL_ORDER = {
    "s": ["s"],
    "p": ["pz", "px", "py"],
    "d": ["dz2", "dzx", "dzy", "dx2-y2", "dxy"],
    "f": ["fz3", "fxz2", "fyz2", "fx( x2-3y2 )", "fxyz", "fy(3x2-y2)", "fx3"]
}
CONVENTIONAL_CUBIC_ORBITAL_ORDER = {
    "s": ["s"],
    "p": ["py", "pz", "px"],
    "d": ["dxy", "dyz", "dz2", "dxz", "x2-y2"],
    "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]
}


# Canonical non-collinear (SOC) ordering
NONCOLINEAR_AZIMUTHAL_ORBITAL_ORDER = [
    {"l": 0, "j": 0.5, "m_j": -0.5},
    {"l": 0, "j": 0.5, "m_j": 0.5},
    {"l": 1, "j": 0.5, "m_j": -0.5},
    {"l": 1, "j": 0.5, "m_j": 0.5},
    {"l": 1, "j": 1.5, "m_j": -1.5},
    {"l": 1, "j": 1.5, "m_j": -0.5},
    {"l": 1, "j": 1.5, "m_j": 0.5},
    {"l": 1, "j": 1.5, "m_j": 1.5},
    {"l": 2, "j": 1.5, "m_j": -1.5},
    {"l": 2, "j": 1.5, "m_j": -0.5},
    {"l": 2, "j": 1.5, "m_j": 0.5},
    {"l": 2, "j": 1.5, "m_j": 1.5},
    {"l": 2, "j": 2.5, "m_j": -2.5},
    {"l": 2, "j": 2.5, "m_j": -1.5},
    {"l": 2, "j": 2.5, "m_j": -0.5},
    {"l": 2, "j": 2.5, "m_j": 0.5},
    {"l": 2, "j": 2.5, "m_j": 1.5},
    {"l": 2, "j": 2.5, "m_j": 2.5},
    {"l": 3, "j": 2.5, "m_j": -2.5},
    {"l": 3, "j": 2.5, "m_j": -1.5},
    {"l": 3, "j": 2.5, "m_j": -0.5},
    {"l": 3, "j": 2.5, "m_j": 0.5},
    {"l": 3, "j": 2.5, "m_j": 1.5},
    {"l": 3, "j": 2.5, "m_j": 2.5},
    {"l": 3, "j": 3.5, "m_j": -3.5},
    {"l": 3, "j": 3.5, "m_j": -2.5},
    {"l": 3, "j": 3.5, "m_j": -1.5},
    {"l": 3, "j": 3.5, "m_j": -0.5},
    {"l": 3, "j": 3.5, "m_j": 0.5},
    {"l": 3, "j": 3.5, "m_j": 1.5},
    {"l": 3, "j": 3.5, "m_j": 2.5},
    {"l": 3, "j": 3.5, "m_j": 3.5},
    # Extend for f orbitals with SOC if needed
]

@dataclass
class OrbitalOrdering:
    azimuthal_order: Dict[str, List[str]] = field(default_factory=lambda: AZIMUTHAL_ORBITAL_ORDER)
    conventional_order: Dict[str, List[str]] = field(default_factory=lambda: CONVENTIONAL_CUBIC_ORBITAL_ORDER)
    flat_soc_order: List[Dict[str, Union[int, float]]] = field(default_factory=lambda: NONCOLINEAR_AZIMUTHAL_ORBITAL_ORDER)
        
    @property
    def flat_azimuthal(self) -> List[str]:
        return self._flatten_order(self.azimuthal_order)
    
    @property
    def flat_conventional(self) -> List[str]:
        return self._flatten_order(self.conventional_order)
    
    @property
    def az_to_conv_map(self) -> Dict[int, int]:
        return self._build_index_map(self.flat_azimuthal, self.flat_conventional)
    
    @property
    def conv_to_az_map(self) -> Dict[int, int]:
        return {v: k for k, v in self.az_to_conv_map.items()}
    
    @property
    def az_to_flat_index(self) -> Dict[int, int]:
        return {orbital_name: i for i, orbital_name in enumerate(self.flat_azimuthal)}
    
    @property
    def conv_to_flat_index(self) -> Dict[int, int]:
        return {orbital_name: i for i, orbital_name in enumerate(self.flat_conventional)}
    
    @property
    def l_orbital_map(self) -> Dict[str, int]:
        return {l_orbital_name: i for i, l_orbital_name in enumerate(self.azimuthal_order.keys())}
    
    @property
    def az_to_lm_records(self) -> List[Dict[str, int]]:
        az_to_lm_records = []
        for l_orbital_name in self.azimuthal_order.keys():
            for i_m, m_orbital_name in enumerate(self.azimuthal_order[l_orbital_name]):
                az_to_lm_records.append({
                    "l": self.l_orbital_map[l_orbital_name],
                    "m": i_m+1,
                })
        return az_to_lm_records
    

    def _flatten_order(self, order_dict: Dict[str, List[str]]) -> List[str]:
        """Flatten orbital dictionary into a single list in order."""
        flat_list = []
        for l_type in ["s", "p", "d", "f"]:
            if l_type in order_dict:
                flat_list.extend(order_dict[l_type])
        return flat_list

    def _build_index_map(
        self, list_a: List[str], list_b: List[str]
    ) -> Dict[int, int]:
        """Map indices from list_a to list_b."""
        mapping = {}
        for i, orb in enumerate(list_a):
            if orb in list_b:
                mapping[i] = list_b.index(orb)
        return mapping

    def get_index(
        self, orbital: str, convention: str = "azimuthal"
    ) -> int:
        """Get index of orbital in chosen convention."""
        if convention == "azimuthal":
            return self.flat_azimuthal.index(orbital)
        elif convention == "conventional":
            return self.flat_conventional.index(orbital)
        else:
            raise ValueError("Convention must be 'azimuthal' or 'conventional'.")

    def to_latex(self, orbital: str) -> str:
        """Convert orbital name to LaTeX math string."""
        replacements = {
            "pz": "p_z",
            "px": "p_x",
            "py": "p_y",
            "dz2": "d_{z^2}",
            "dzx": "d_{zx}",
            "dzy": "d_{zy}",
            "dx2-y2": "d_{x^2-y^2}",
            "dxy": "d_{xy}",
            "fz3": "f_{z^3}",
            "fxz2": "f_{xz^2}",
            "fyz2": "f_{yz^2}",
            "fx( x2-3y2 )": "f_{x(x^2-3y^2)}",
            "fxyz": "f_{xyz}",
            "fy(3x2-y2)": "f_{y(3x^2-y^2)}",
            "fx3": "f_{x^3}",
        }
        return f"$\\{replacements.get(orbital, orbital)}$"

    def map_index(
        self, index: int, from_convention: str, to_convention: str
    ) -> int:
        """Map index between conventions."""
        if from_convention == "azimuthal" and to_convention == "conventional":
            return self.az_to_conv_map[index]
        elif from_convention == "conventional" and to_convention == "azimuthal":
            return self.conv_to_az_map[index]
        else:
            raise ValueError("Invalid convention mapping.")

    def get_soc_index(self, l: int, j: float, m: float) -> int:
        """Get index in SOC ordering."""
        for idx, entry in enumerate(self.flat_soc_order):
            if entry["l"] == l and entry["j"] == j and entry["m_j"] == m:
                return idx
        raise ValueError("SOC orbital not found.")




elements = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109}

