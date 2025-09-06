__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import ast
import copy
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io.qe.utils import (
    QECardBlock,
    QEScalar,
    QETensor,
    QEValue,
    convert_to_typed_value,
    parse_qe_input_cards,
)
from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")




def _convert_lorbnum_to_letter(lorbnum: int) -> str:
    mapping = {0: "s", 1: "p", 2: "d", 3: "f"}
    return mapping[lorbnum]

def str2bool(v):
    """Converts a string of a boolean to an actual boolean

    Parameters
    ----------
    v : str
        The string of the boolean value

    Returns
    -------
    boolean
        The boolean value
    """
    return v.lower() in ("true")


    
# ===== Specialized Subclasses =====
@dataclass
class ControlCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
    
@dataclass
class SystemCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
    
    
@dataclass
class ElectronsCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
    
    
@dataclass
class IonsCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
     
    
@dataclass
class CellCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
     
@dataclass
class FCPCard(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
    
    
@dataclass
class RISM(QECardBlock):
    def __post_init__(self) -> None:
        self.parse()
        
    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block) 
    
    @property
    def data(self) -> Dict[str, str]:
        return self._data
     
    
@dataclass
class AtomicSpeciesCard(QECardBlock):
    labels: list[str] = field(default_factory=list)
    masses: Optional[np.ndarray] = None
    pseudopotentials: list[str] = field(default_factory=list)
    pseudo_formats: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parse()

    @staticmethod
    def _infer_pseudo_format(filename: str) -> str:
        name = filename.strip()
        low = name.lower()
        if low.endswith(".vdb") or low.endswith(".van"):
            return "USPP"
        if low.endswith(".rrkj3"):
            return "RRKJ3"
        if low.endswith(".upf"):
            return "UPF"
        return "NC"

    @staticmethod
    def _validate_label(label: str) -> bool:
        if not (1 <= len(label) <= 3):
            return False
        # Accept 1-2 letters optionally followed by one alnum or _ or - and one alnum
        # Examples: C, Si, Fe1, C_h, C-h
        if re.fullmatch(r"[A-Za-z]{1,2}", label):
            return True
        if re.fullmatch(r"[A-Za-z]{1,2}[0-9A-Za-z]", label):
            return True
        if re.fullmatch(r"[A-Za-z]{1,2}[_-][0-9A-Za-z]", label):
            return True
        return False

    def parse(self) -> None:
        raw_lines = self.block.splitlines() if self.block else []
        lines: list[str] = []
        for line in raw_lines:
            cleaned = line.split("!", 1)[0].split("#", 1)[0].strip()
            if cleaned:
                lines.append(cleaned)

        labels: list[str] = []
        masses: list[float] = []
        pseudos: list[str] = []
        formats: list[str] = []

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            label, mass_str, pseudo = parts[0], parts[1], parts[2]
            if not self._validate_label(label):
                # Still accept but log debug; keep strictness minimal
                logger.debug("Atomic species label '%s' did not match validator", label)
            try:
                m = float(mass_str.replace('D', 'E').replace('d', 'e'))
            except Exception:
                # Skip invalid mass lines
                continue
            labels.append(label)
            masses.append(m)
            pseudos.append(pseudo)
            formats.append(self._infer_pseudo_format(pseudo))

        self.labels = labels
        self.masses = np.array(masses, dtype=float) if masses else None
        self.pseudopotentials = pseudos
        self.pseudo_formats = formats

    @cached_property
    def species(self) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        if self.labels and self.masses is not None:
            for lbl, m in zip(self.labels, self.masses.tolist()):
                mapping[lbl] = m
        return mapping

@dataclass
class AtomicPositionsCard(QECardBlock):
    # Parsed attributes
    mode: str | None = None  # one of: alat, bohr, angstrom, crystal, crystal_sg
    labels: list[str] = field(default_factory=list)
    positions: np.ndarray | None = None    # shape (nat, 3) when 3 coords provided
    constraints: np.ndarray | None = None  # shape (nat, 3) integers (0/1)
    wyckoff: list[str] = field(default_factory=list)  # for crystal_sg
    wyckoff_params: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parse()

    @staticmethod
    def _evaluate_expr(token: str) -> float:
        """Safely evaluate a simple arithmetic expression used by QE.

        Supports +, -, *, /, ^ (power), parentheses, and unary minus.
        Disallows names, calls, or any other Python features.
        """
        expr = token.strip()
        if not expr:
            raise ValueError("Empty expression")
        # QE uses '^' for power
        expr = expr.replace("^", "**")
        # Disallow leading '+' as per QE note
        if expr.startswith('+'):
            raise ValueError("Leading '+' not allowed in QE expressions")

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Num,  # for older Python ASTs
            ast.Tuple,  # not expected but harmless if encountered
            ast.Load,
            ast.Mod,  # not used, but reject at validation below
        )

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):  # type: ignore[attr-defined]
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("Invalid constant in expression")
            if isinstance(node, ast.Num):  # py<3.8
                return float(node.n)  # type: ignore[attr-defined]
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    return -_eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    # QE disallows leading '+', treat as invalid
                    raise ValueError("Leading '+' not allowed in QE expressions")
                raise ValueError("Unsupported unary operator")
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                raise ValueError("Unsupported binary operator")
            # Reject all other nodes (Names, Calls, etc.)
            raise ValueError("Unsupported expression element")

        tree = ast.parse(expr, mode='eval')
        for n in ast.walk(tree):
            if not isinstance(n, allowed_nodes):
                raise ValueError("Disallowed token in expression")
        return float(_eval(tree))

    @staticmethod
    def _parse_if_pos(raw: str) -> tuple[int, int, int]:
        tmp = raw.replace('{', ' ').replace('}', ' ').strip()
        parts = [p for p in re.split(r"[\s,]+", tmp) if p]
        vals: list[int] = []
        for p in parts[:3]:
            try:
                vals.append(int(float(p)))
            except Exception:
                vals.append(1)
        while len(vals) < 3:
            vals.append(1)
        return vals[0], vals[1], vals[2]

    def parse(self) -> None:
        # Normalize mode from options, remove braces and parentheses
        mode_raw = (self.options or "").strip().lower()
        mode_raw = mode_raw.replace('{', '').replace('}', '')
        mode_raw = mode_raw.replace('(', '').replace(')', '')
        self.mode = mode_raw if mode_raw else "alat"
        if self.mode not in {"alat", "bohr", "angstrom", "crystal", "crystal_sg"}:
            self.mode = "alat"

        raw_lines = self.block.splitlines() if self.block else []
        lines: list[str] = []
        for line in raw_lines:
            cleaned = line.split("!", 1)[0].split("#", 1)[0].strip()
            if cleaned:
                lines.append(cleaned)

        labels: list[str] = []
        coords: list[tuple[float, float, float]] = []
        constr: list[tuple[int, int, int]] = []
        wyckoff_list: list[Optional[str]] = []
        wyckoff_params_list: list[tuple[Optional[float], Optional[float], Optional[float]]] = []

        wyckoff_re = re.compile(r"^[0-9]+[A-Za-z]+$")

        for line in lines:
            # Split off optional constraints in braces
            if '{' in line and '}' in line:
                line_part, brace_part = line.split('{', 1)
                brace_content = '{' + brace_part
                ifpos = self._parse_if_pos(brace_content)
            else:
                line_part = line
                ifpos = (1, 1, 1)

            tokens = [t for t in line_part.split() if t]
            if not tokens:
                continue
            label = tokens[0]

            # By default assume three coordinate tokens follow
            rest = tokens[1:]

            # crystal_sg: allow wyckoff label then up to 3 params
            if self.mode == "crystal_sg" and rest:
                first = rest[0]
                if wyckoff_re.match(first):
                    wy = first
                    param_tokens = rest[1:]
                    # up to 3 params, may be expressions
                    pvals: list[Optional[float]] = []
                    for tk in param_tokens[:3]:
                        try:
                            pvals.append(self._evaluate_expr(tk))
                        except Exception:
                            pvals.append(None)
                    while len(pvals) < 3:
                        pvals.append(None)
                    labels.append(label)
                    wyckoff_list.append(wy)
                    wyckoff_params_list.append((pvals[0], pvals[1], pvals[2]))
                    # Positions are not expanded for crystal_sg without full symmetry; leave as NaN
                    coords.append((np.nan, np.nan, np.nan))
                    constr.append(ifpos)
                    continue

            # Coordinates provided (alat/bohr/angstrom/crystal, or crystal_sg without wyckoff)
            if len(rest) < 3:
                # Not enough tokens for coordinates; skip
                continue
            try:
                x = self._evaluate_expr(rest[0])
                y = self._evaluate_expr(rest[1])
                z = self._evaluate_expr(rest[2])
            except Exception:
                # Skip invalid line
                continue

            labels.append(label)
            coords.append((x, y, z))
            constr.append(ifpos)
            wyckoff_list.append(None)
            wyckoff_params_list.append((None, None, None))

        self.labels = labels
        self.positions = np.array(coords, dtype=float) if coords else None
        self.constraints = np.array(constr, dtype=int) if constr else None
        self.wyckoff = wyckoff_list
        self.wyckoff_params = wyckoff_params_list

@dataclass
class KPointsCard(QECardBlock):
    mode: str | None = None
    nks: int | None = None
    kpoints: np.ndarray | None = None
    weights: np.ndarray | None = None
    line_points: list[int] = field(default_factory=list)
    line_comments: list[str] = field(default_factory=list)
    nk1: int | None = None
    nk2: int | None = None
    nk3: int | None = None
    sk1: int | None = None
    sk2: int | None = None
    sk3: int | None = None
    is_gamma: bool = False
    knames: list[str] = field(default_factory=list)
    kticks: list[int] = field(default_factory=list)
    nhigh_sym: int | None = None
    ngrids: list[int] = field(default_factory=list)
    high_symmetry_points: np.ndarray | None = None
    special_kpoints: np.ndarray | None = None
    modified_knames: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> "KPointsCard.KPointsInfo":
        """Parse the K_POINTS card body according to its option/mode.

        Returns
        ----
        KPointsInfo
            Parsed K_POINTS information.
        """
        mode_raw = (self.options or "").strip().lower()
        self.mode = mode_raw if mode_raw else "tpiba"

        # Normalize accepted modes
        valid_modes = {
            "tpiba",
            "automatic",
            "crystal",
            "gamma",
            "tpiba_b",
            "crystal_b",
            "tpiba_c",
            "crystal_c",
        }
        if self.mode not in valid_modes:
            # Fallback to default behavior (tpiba) if unspecified/unknown
            self.mode = "tpiba"

        # Preprocess lines: strip comments and blanks
        raw_lines = self.block.splitlines() if self.block else []
        lines: list[str] = []
        for line in raw_lines:
            cleaned = line.strip()
            if cleaned:
                lines.append(cleaned)

        # Handle each mode
        if self.mode == "automatic":
            self.parse_automatic_mode(lines)
        elif self.mode == "gamma":
            self.parse_gamma_mode()
            
        elif self.mode == "crystal":
            self.parse_crystal_mode()
        
        elif self.mode == "crystal_b":
            self.parse_crystal_b_mode()
            
        else:
            self.parse_explicit_mode(lines)
        return 
    
    def parse_automatic_mode(self, lines: list[str]) -> None:
        if not lines:
            return None
        parts = lines[0].split()
        if len(parts) < 6:
            return None
        nk1, nk2, nk3, sk1, sk2, sk3 = [int(float(x)) for x in parts[:6]]
        self.nk1 = nk1
        self.nk2 = nk2
        self.nk3 = nk3
        self.sk1 = sk1
        self.sk2 = sk2
        self.sk3 = sk3
        return None

    def parse_gamma_mode(self) -> None:
        self.kpoints = np.zeros((1, 3), dtype=float)
        self.weights = np.array([1.0], dtype=float)
        self.is_gamma = True
        return None
    
    def parse_crystal_mode(self) -> None:
        lines = self.block.splitlines()   
        n_kpoints = int(lines[0])
        self.knames = []
        self.kticks = []
        self.line_comments = []

        for itick, x in enumerate(lines[1:]):
            cols = x.split()
            if len(cols) == 5:
                comment = cols[4].strip()
                k_name = comment.replace("!", "").replace("#", "").strip()
                self.knames.append(k_name)
                self.kticks.append(itick)
                self.line_comments.append(comment)
        self.nhigh_sym = len(self.knames)
        
    def parse_crystal_b_mode(self) -> None:
        lines = self.block.splitlines()
        self.nhigh_sym = int(lines[0])
        high_symmetry_points = []
        line_points = []
        for line in lines[1:]:
            if line.strip():
                cols = line.split()
                kx,ky,kz,n_points = cols[:4]
                high_symmetry_points.append([float(kx), float(ky), float(kz)])
                line_points.append(int(n_points))
                
                if len(cols) == 5:
                    comment = cols[4].strip().replace("!", "").replace("#", "").strip()
                    self.line_comments.append(comment)
                else:
                    self.line_comments.append("")
        
        self.high_symmetry_points = np.array(high_symmetry_points, dtype=float)
        self.line_points = np.array(line_points, dtype=int)
        self.kticks = []

        tick_Count = 1
        for ihs in range(self.nhigh_sym):

            # In QE cyrstal_b mode, the user is able to specify grid on last high symmetry point.
            # QE just uses 1 for the last high symmetry point.
            grid_current = self.line_points[ihs]
            if ihs < self.nhigh_sym - 2:
                self.ngrids.append(grid_current)

            # Incrementing grid by 1 for seocnd to last high symmetry point
            elif ihs == self.nhigh_sym - 2:
                self.ngrids.append(grid_current + 1)

            # I have no idea why I skip the last high symmetry point. I think it had to do with disconinuous points.
            # Need to code test case for this. Otherwise leave it as is.
            # elif ihs == self.nhigh_sym - 1:
            #     continue
            self.kticks.append(tick_Count - 1)
            tick_Count += grid_current

        # Initial guess for knames
        self.knames = [str(x) for x in range(self.nhigh_sym)]
        if len(self.line_comments) == self.nhigh_sym:
            tmp_knames = []
            for i, comment in enumerate(self.line_comments):
                tmp_knames.append(comment.replace(",", "").replace("vlvp1d", "").replace(" ", ""))
            self.knames = tmp_knames
        
        # Formating to conform with Kpath class
        self.special_kpoints = np.zeros(shape=(len(self.kticks) - 1, 2, 3))

        self.modified_knames = []
        for itick in range(len(self.kticks)):
            if itick != len(self.kticks) - 1:
                self.special_kpoints[itick, 0, :] = self.high_symmetry_points[itick]
                self.special_kpoints[itick, 1, :] = self.high_symmetry_points[itick + 1]
                self.modified_knames.append(
                    [self.knames[itick], self.knames[itick + 1]]
                )
    
    def parse_explicit_mode(self, lines: list[str]) -> None:
        # All other explicit-list modes expect: first line is nks, followed by nks lines
        if not lines:
            return None
        try:
            nks = int(float(lines[0].split()[0]))
        except Exception:
            nks = 0
        kpts_list: list[list[float]] = []
        wts_list: list[float] = []
        line_comments: list[str] = []
        for i in range(1, min(1 + nks, len(lines))):
            cols = lines[i].split()
            if len(cols) < 4:
                # Some inputs might provide only kx ky kz without weight; default to 1.0
                try:
                    kx, ky, kz = map(float, cols[:3])
                    wt = 1.0
                    label = ""
                except Exception:
                    continue
            elif len(cols) == 4:
                try:
                    kx, ky, kz, wt = map(float, cols[:4])
                    label = ""
                except Exception:
                    # Last-column could be a label; try first three as k, last numeric as weight
                    try:
                        kx, ky, kz = map(float, cols[:3])
                        wt = float(cols[3])
                        label = ""
                    except Exception:
                        continue
            elif len(cols) == 5:
                try:
                    kx, ky, kz, wt = map(float, cols[:4])
                    label = cols[-1].replace("!", "").replace("#", "")
                except Exception:
                    continue
            else:
                continue
            kpts_list.append([kx, ky, kz])
            wts_list.append(wt)
            line_comments.append(label)

        if nks and len(kpts_list) != nks:
            # If fewer lines were parsed than declared, adjust to what we have
            nks = len(kpts_list)

        kpts_arr = np.array(kpts_list, dtype=float) if kpts_list else None
        wts_arr = np.array(wts_list, dtype=float) if wts_list else None
        
        self.kpoints=kpts_arr
        self.weights=wts_arr
        self.line_comments=line_comments
    
 
@dataclass
class OccupationsCard(QECardBlock):
    @cached_property
    def data(self) -> Dict[str, str]:
        return parse_qe_input_cards(self.block) 

@dataclass
class ConstraintsCard(QECardBlock):
    @cached_property
    def data(self) -> Dict[str, str]:
        return parse_qe_input_cards(self.block) 

@dataclass
class AdditionalKPointsCard(QECardBlock):
    @cached_property
    def data(self) -> Dict[str, str]:
        return parse_qe_input_cards(self.block) 

@dataclass
class SolventsCard(QECardBlock):
    @cached_property
    def data(self) -> Dict[str, str]:
        return parse_qe_input_cards(self.block) 

@dataclass
class HubbardCard(QECardBlock):
    @cached_property    
    def data(self) -> Dict[str, str]:
        return parse_qe_input_cards(self.block) 
    
class QECardBlockEnum(Enum):
    control = ControlCard
    system = SystemCard
    electrons = ElectronsCard
    atomic_species = AtomicSpeciesCard
    atomic_positions = AtomicPositionsCard
    k_points = KPointsCard
    cell_parameters = CellCard
    occupations = OccupationsCard
    constraints = ConstraintsCard
    additional_k_points = AdditionalKPointsCard
    solvents = SolventsCard
    hubbard = HubbardCard
    unknown = QECardBlock
    
    @classmethod
    def from_name(cls, name: str) -> "QECardBlockEnum":
        return cls[name.lower()]
    
    @classmethod
    def keys(cls) -> List[str]:
        return list(cls.__members__.keys())
    
    @classmethod
    def is_in(cls, name: str) -> bool:
        return name.lower() in cls.__members__.keys()

# ===== Parsing / Extraction =====
def extract_qe_input_blocks(text: str) -> List[QECardBlock]:
    """
    Extract QE namelist (&.../) and card blocks from text.

    Returns:
        List of QECardBlock in the order they appear.
    """
    text = text.replace("\r\n", "\n")

    # Pattern for namelist: &NAME ... /
    namelist_pattern = (
        r"^&(?P<nl_name>[A-Za-z0-9_]+)\s*\n"
        r"(?P<nl_body>[\s\S]*?)^\s*/\s*$"
    )

    # Pattern for card: NAME {options} + block
    # Restrict NAME to known QE card headers to avoid confusing species lines
    known_cards = [
        name for name in QECardBlockEnum.keys() if name not in ("unknown",)
    ]
    known_headers = "|".join(re.escape(n.upper()) for n in known_cards)
    card_pattern = (
        rf"^(?P<card_name>(?:{known_headers}))"
        rf"(?:[ \t]+(?P<card_options>(?:\{{[^}}]*\}}|[^\n]+)))?"  # options only on same line
        rf"[ \t]*\n"
        rf"(?P<card_body>(?:(?!^(?:{known_headers})\b|^&).*(?:\n|$))*)"
    )

    combined_pattern = f"(?:{namelist_pattern})|(?:{card_pattern})"

    blocks: List[QECardBlock] = []
    for match in re.finditer(combined_pattern, text, re.MULTILINE):
        if match.group("nl_name"):
            name = match.group("nl_name").strip()
            options = ""
            block = match.group("nl_body").strip("\n ")
        elif match.group("card_name"):
            name = match.group("card_name").strip()
            opt = match.group("card_options") or ""
            options = opt.strip()
            if options.startswith("{") and options.endswith("}"):
                options = options[1:-1].strip()
            block = match.group("card_body").strip("\n ")
        else:
            raise ValueError(f"Error with parsing QE input blocks: Invalid card or namelist: {match.group()}")
        if QECardBlockEnum.is_in(name):
            blocks.append(QECardBlockEnum.from_name(name).value(name, options, block))
        else:
            blocks.append(QECardBlockEnum["unknown"].value(name, options, block))

    return blocks


class PwIn:
    """Parser for the input of the PW module in Quantum ESPRESSO with structured sections (no dataclass)."""
    
    @classmethod
    def is_file_of_type(cls, filepath: Union[str, Path]) -> bool:
        """Quickly determine if an input file looks like a PWSCF input.

        Checks the beginning of the file for both '&control' and '&system'
        namelists (case-insensitive).
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                head = "".join([f.readline() for _ in range(50)])
            if not head:
                return False
            return (
                re.search(r"^.*&control\b", head, re.IGNORECASE | re.MULTILINE) is not None
                and re.search(r"^.*&system\b", head, re.IGNORECASE | re.MULTILINE) is not None
            )
        except Exception:
            return False

    def __init__(self,filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()

    def _read(self) -> str:
        with open(self.filepath, "r") as f:
            return f.read()
    
    @cached_property
    def text(self) -> str:
        return self._text
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @cached_property
    def data(self) -> List[QECardBlock]:
        data={}
        qe_card_blocks = extract_qe_input_blocks(self.text)
        return {block.name: block for block in qe_card_blocks}
    
    @cached_property
    def control_card(self) -> ControlCard:
        for block in self.data:
            if isinstance(block, ControlCard):
                return block
        raise ValueError("ControlCard not found in PWInput")
    
    @cached_property
    def system_card(self) -> SystemCard:
        for block in self.data:
            if isinstance(block, SystemCard):
                return block
        raise ValueError("SystemCard not found in PWInput")
    
    @cached_property
    def electrons_card(self) -> ElectronsCard:
        for block in self.data:
            if isinstance(block, ElectronsCard):
                return block
        raise ValueError("ElectronsCard not found in PWInput")
    
    @cached_property
    def atomic_species_card(self) -> AtomicSpeciesCard:
        for block in self.data:
            if isinstance(block, AtomicSpeciesCard):
                return block
        raise ValueError("AtomicSpeciesCard not found in PWInput")
    
    @cached_property
    def atomic_positions_card(self) -> AtomicPositionsCard:
        for block in self.data:
            if isinstance(block, AtomicPositionsCard):
                return block
        raise ValueError("AtomicPositionsCard not found in PWInput")
    
    @cached_property
    def kpoints_card(self) -> KPointsCard:
        for card_name, card in self.data.items():
            if isinstance(card, KPointsCard):
                return card
        raise ValueError("KPointsCard not found in PWInput")
    
    @cached_property
    def cell_card(self) -> CellCard:
        for block in self.data:
            if isinstance(block, CellCard):
                return block
        raise ValueError("CellCard not found in PWInput")
    
    @cached_property
    def occupations_card(self) -> OccupationsCard:
        for block in self.data:
            if isinstance(block, OccupationsCard):
                return block
        raise ValueError("OccupationsCard not found in PWInput")

    @cached_property
    def bands_kpoint_names(self) -> Optional[list[str]]:
        """Extract k-point labels from the K_POINTS card for bands runs.

        This parses labels specified after a '!' comment (e.g., "... !Gamma")
        or as a trailing token, handling both ``K_POINTS crystal`` and
        ``K_POINTS crystal_b`` modes.

        Returns
        ----
        list of str or None
            Ordered list of labels for the high-symmetry points if
            ``calculation = 'bands'`` and labels are present; otherwise None.
        """
        try:
            blocks = extract_qe_input_blocks(self.text)
        except Exception:
            return None

        # Detect calculation type
        calc_mode: Optional[str] = None
        for b in blocks:
            if isinstance(b, ControlCard):
                data = b.data
                calc_mode = (data.get("calculation") if isinstance(data, dict) else None)
                if isinstance(calc_mode, str):
                    calc_mode = calc_mode.strip().lower().strip("'\"")
                break

        if calc_mode != "bands":
            return None

        # Find K_POINTS block
        kpoints_block: Optional[KPointsCard] = None
        for b in blocks:
            if isinstance(b, KPointsCard):
                kpoints_block = b
                break
        if kpoints_block is None:
            return None

        mode = (kpoints_block.options or "").strip().lower() if hasattr(kpoints_block, "options") else ""
        if mode not in {"crystal", "crystal_b", "tpiba", "tpiba_b", "crystal_c", "tpiba_c", "gamma"}:
            # Unknown or unsupported for band labels
            return None

        raw = (kpoints_block.block or "") if hasattr(kpoints_block, "block") else ""
        if not raw:
            return None

        raw_lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not raw_lines:
            return None

        def _label_from_line(line: str) -> Optional[str]:
            # Prefer comment-based labels after '!'
            if "!" in line:
                return line.split("!", 1)[1].strip().replace(",", "").replace(" ", "")
            # Else, attempt to read a 5th token as label (rare)
            parts = line.split()
            if len(parts) >= 5:
                return parts[4].strip().replace(",", "").replace(" ", "")
            return None

        # In explicit modes, first non-empty line is typically count of points
        labels: list[str] = []
        try:
            n_declared = int(float(raw_lines[0].split()[0]))
            point_lines = raw_lines[1:1 + n_declared]
        except Exception:
            # If not declared, assume all lines are k-point entries
            point_lines = raw_lines

        for ln in point_lines:
            lbl = _label_from_line(ln)
            if lbl is None or lbl == "":
                # Keep place with numeric index if unlabeled
                labels.append(str(len(labels)))
            else:
                labels.append(lbl)

        return labels or None



class PwOut:
    """Lightweight parser for the QE scf.out file."""

    @classmethod
    def is_file_of_type(cls, filepath: Union[str, Path]) -> bool:
        """Quickly determine if a .out file belongs to PWSCF.

        Reads up to the first 5 lines and checks for 'PWSCF' (case-insensitive).
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    if re.search(r"PWSCF", line, re.IGNORECASE):
                        return True
        except Exception:
            pass
        return False

    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()

    def _read(self) -> str:
        with open(self.filepath, "r") as f:
            return f.read()
        
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @cached_property
    def text(self) -> str:
        return self._text
    
    @cached_property
    def version_tuple(self) -> tuple[int, int, int] | None:
        pattern = re.compile(r"^\s*Program\s+PROJWFC\s+v\.?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", re.I | re.M)
        m = pattern.search(self.text)
        if m:
            version_tuple = list(int(g) for g in m.groups() if g is not None)
            if len(version_tuple) == 2:
                version_tuple.append(0)
            version_tuple = tuple(version_tuple)
        else:
            version_tuple = None
            
        return version_tuple
    
    @cached_property
    def version(self) -> str | None:
        if self.version_tuple:
            return f"{self.version_tuple[0]}.{self.version_tuple[1]}.{self.version_tuple[2]}"
        else:
            return None
        
    @cached_property
    def _parallel_info(self) -> dict[str, Any]:
        """Parse parallel execution information from ``projwfc.out``."""
        info: dict[str, Any] = {
            "parallel_version": None,
            "n_cores": None,
            "mpi_processes": None,
            "n_threads": None,
            "n_nodes": None,
            "n_pool": None,
            "proc_nbgrp_npool_nimage": None,
            "available_mem": None,
        }
        text = self.text

        # Example: Parallel version (MPI & OpenMP), running on     112 processor cores
        m = re.search(
            r"^\s*Parallel\s+version\s*\(\s*([^)]+?)\s*\)\s*,\s*running\s+on\s+(\d+).*",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["parallel_version"] = m.group(1).strip()
            info["n_cores"] = int(m.group(2))

        # Example: Number of MPI processes:               112
        m = re.search(r"^\s*Number\s+of\s+MPI\s+processes:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["mpi_processes"] = int(m.group(1))

        # Example: Threads/MPI process:                     1
        m = re.search(r"^\s*Threads/MPI\s+process:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_threads"] = int(m.group(1))

        # Example: MPI processes distributed on     1 nodes
        m = re.search(r"^\s*MPI\s+processes\s+distributed\s+on\s+(\d+)\s+nodes", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_nodes"] = int(m.group(1))

        # Example: K-points division:     npool     =       7
        m = re.search(r"^\s*K-points\s+division:\s*npool\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_pool"] = int(m.group(1))

        # Example (variants):
        #   R & G space division:  proc/nbgrp/npool/nimage =  16/  1/  8/  1
        #   R & G space division:  proc/nbgrp/npool/nimage =      16
        m = re.search(
            r"^\s*R\s*&\s*G\s*space\s*division:\s*proc/nbgrp/npool/nimage\s*=\s*([0-9\s/]+)",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            info["proc_nbgrp_npool_nimage"] = nums  # length can be 1 or 4

        # Example: 469343 MiB available memory ...
        m = re.search(r"^\s*(\d+)\s*MiB\s+available\s+memory", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["available_mem"] = int(m.group(1))

        return info
    
    @cached_property
    def parallel_version(self) -> str | None:
        """Parallel build description, e.g., 'MPI & OpenMP'."""
        return self._parallel_info["parallel_version"]

    @cached_property
    def n_cores(self) -> int | None:
        """Number of processor cores reported."""
        return self._parallel_info["n_cores"]

    @cached_property
    def mpi_processes(self) -> int | None:
        """Number of MPI processes."""
        return self._parallel_info["mpi_processes"]

    @cached_property
    def n_threads(self) -> int | None:
        """Threads per MPI process."""
        return self._parallel_info["n_threads"]

    @cached_property
    def n_nodes(self) -> int | None:
        """Number of nodes used."""
        return self._parallel_info["n_nodes"]

    @cached_property
    def n_pool(self) -> int | None:
        """npool for k-point division."""
        return self._parallel_info["n_pool"]

    @cached_property
    def proc_nbgrp_npool_nimage(self) -> list[int] | None:
        """proc/nbgrp/npool/nimage numbers as a list (length 1 or 4)."""
        return self._parallel_info["proc_nbgrp_npool_nimage"]


    @cached_property
    def _parallelization_details(self) -> dict[str, Any] | None:
        """Parse 'Parallelization info' table from QE outputs.

        Returns
        ----
        dict or None
            A dict with keys 'sticks' and 'gvecs', each mapping to a dict
            of per-channel 'dense'/'smooth'/'pw' stats, with 'min'/'max'/'sum'.

            Example:
            {
              "sticks": {
                "dense": {"min": 48, "max": 49, "sum": 969},
                "smooth": {"min": 24, "max": 25, "sum": 483},
                "pw": {"min": 8, "max": 9, "sum": 173},
              },
              "gvecs": {
                "dense": {"min": 956, "max": 958, "sum": 19141},
                "smooth": {"min": 340, "max": 343, "sum": 6819},
                "pw": {"min": 73, "max": 77, "sum": 1505},
              },
            }
        """
        text = self.text
        # Anchor on the section header
        header = re.search(r"^\s*Parallelization\s+info\s*$", text, re.IGNORECASE | re.MULTILINE)
        if not header:
            return None

        # Collect the next few lines to find Min/Max/Sum rows
        after = text[header.end():]

        # Each row has three numbers for sticks and three for G-vecs
        row_re = re.compile(
            r"^\s*(Min|Max|Sum)\s+"
            r"(\d+)\s+(\d+)\s+(\d+).*?"        # sticks: dense, smooth, pw
            r"(\d+)\s+(\d+)\s+(\d+)\s*$",      # g-vecs: dense, smooth, pw
            re.IGNORECASE | re.MULTILINE,
        )

        stats: dict[str, dict[str, dict[str, int]]] = {
            "sticks": {"dense": {}, "smooth": {}, "pw": {}},
            "gvecs": {"dense": {}, "smooth": {}, "pw": {}},
        }

        found = False
        for m in row_re.finditer(after):
            found = True
            label = m.group(1).lower()  # 'min' | 'max' | 'sum'
            s_dense, s_smooth, s_pw = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            g_dense, g_smooth, g_pw = (int(m.group(5)), int(m.group(6)), int(m.group(7)))

            stats["sticks"]["dense"][label] = s_dense
            stats["sticks"]["smooth"][label] = s_smooth
            stats["sticks"]["pw"][label] = s_pw

            stats["gvecs"]["dense"][label] = g_dense
            stats["gvecs"]["smooth"][label] = g_smooth
            stats["gvecs"]["pw"][label] = g_pw

        return stats if found else None

    @cached_property
    def parallelization_table(self) -> dict[str, Any] | None:
        """Structured 'Parallelization info' stats if present."""
        return self._parallelization_details

    @cached_property
    def sticks_min(self) -> tuple[int, int, int] | None:
        """Min sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("min"),
            d["sticks"]["smooth"].get("min"),
            d["sticks"]["pw"].get("min"),
        )

    @cached_property
    def sticks_max(self) -> tuple[int, int, int] | None:
        """Max sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("max"),
            d["sticks"]["smooth"].get("max"),
            d["sticks"]["pw"].get("max"),
        )

    @cached_property
    def sticks_sum(self) -> tuple[int, int, int] | None:
        """Sum sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("sum"),
            d["sticks"]["smooth"].get("sum"),
            d["sticks"]["pw"].get("sum"),
        )

    @cached_property
    def gvecs_min(self) -> tuple[int, int, int] | None:
        """Min G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("min"),
            d["gvecs"]["smooth"].get("min"),
            d["gvecs"]["pw"].get("min"),
        )

    @cached_property
    def gvecs_max(self) -> tuple[int, int, int] | None:
        """Max G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("max"),
            d["gvecs"]["smooth"].get("max"),
            d["gvecs"]["pw"].get("max"),
        )

    @cached_property
    def gvecs_sum(self) -> tuple[int, int, int] | None:
        """Sum G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("sum"),
            d["gvecs"]["smooth"].get("sum"),
            d["gvecs"]["pw"].get("sum"),
        )

    @cached_property
    def using_slab_decomposition(self) -> bool:
        """Whether 'Using Slab Decomposition' appears in the output."""
        return bool(re.search(r"^\s*Using\s+Slab\s+Decomposition", self.text, re.IGNORECASE | re.MULTILINE))
    
    # -------------------------
    # System info
    # -------------------------
    @cached_property
    def _system_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        text = self.text

        patterns = {
            "bravais_index": r"bravais-lattice index\s*=\s*(\d+)",
            "alat": r"lattice parameter \(alat\)\s*=\s*([\d.]+)",
            "cell_volume": r"unit-cell volume\s*=\s*([\d.]+)",
            "natoms": r"number of atoms/cell\s*=\s*(\d+)",
            "ntyp": r"number of atomic types\s*=\s*(\d+)",
            "nelectrons": r"number of electrons\s*=\s*([\d.]+)",
            "nkohn_sham": r"number of Kohn-Sham states\s*=\s*(\d+)",
            "ecutwfc": r"kinetic-energy cutoff\s*=\s*([\d.]+)",
            "ecutrho": r"charge density cutoff\s*=\s*([\d.]+)",
            "conv_thr": r"scf convergence threshold\s*=\s*([\d.Ee+-]+)",
            "mixing_beta": r"mixing beta\s*=\s*([\d.]+)",
            "n_scf_steps": r"number of iterations used\s*=\s*(\d+)",
        }

        for key, pat in patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                val = m.group(1)
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
                info[key] = val
            else:
                info[key] = None

        return info
    
    @cached_property
    def bravais_index(self) -> int | None:
        return self._system_info["bravais_index"]

    @cached_property
    def alat(self) -> float | None:
        return self._system_info["alat"]

    @cached_property
    def cell_volume(self) -> float | None:
        return self._system_info["cell_volume"]

    @cached_property
    def natoms(self) -> int | None:
        return self._system_info["natoms"]

    @cached_property
    def ntyp(self) -> int | None:
        return self._system_info["ntyp"]

    @cached_property
    def nelectrons(self) -> float | None:
        return self._system_info["nelectrons"]

    @cached_property
    def nkohn_sham(self) -> int | None:
        return self._system_info["nkohn_sham"]

    @cached_property
    def ecutwfc(self) -> float | None:
        return self._system_info["ecutwfc"]

    @cached_property
    def ecutrho(self) -> float | None:
        return self._system_info["ecutrho"]

    @cached_property
    def conv_thr(self) -> float | None:
        return self._system_info["conv_thr"]

    @cached_property
    def mixing_beta(self) -> float | None:
        return self._system_info["mixing_beta"]

    @cached_property
    def n_scf_steps(self) -> int | None:
        return self._system_info["n_scf_steps"]
    
    # -------------------------
    # Exchange-correlation
    # -------------------------
    @cached_property
    def exchange_correlation(self) -> dict[str, Any] | None:
        """
        Returns:
            {
                "functional": "SLA  PW   PBX  PBC",
                "params": [1, 4, 3, 4, 0, 0, 0]
            }
        """
        m = re.search(
            r"Exchange-correlation=\s*(.+?)\n\s*\(([\d\s]+)\)",
            self.text,
            re.I,
        )
        if m:
            func = m.group(1).strip()
            params = [int(x) for x in m.group(2).split()]
            return {"functional": func, "params": params}
        return None

    # -------------------------
    # celldm values
    # -------------------------
    @cached_property
    def celldm(self) -> dict[str, float] | None:
        """
        Returns:
            {"celldm1": 7.26885, "celldm2": 0.0, ..., "celldm6": 0.0}
        """
        m = re.search(
            r"celldm\(1\)=\s*([\d.]+)\s*celldm\(2\)=\s*([\d.]+)\s*celldm\(3\)=\s*([\d.]+)\s*"
            r"celldm\(4\)=\s*([\d.]+)\s*celldm\(5\)=\s*([\d.]+)\s*celldm\(6\)=\s*([\d.]+)",
            self.text,
            re.I,
        )
        if m:
            vals = [float(x) for x in m.groups()]
            return {f"celldm{i+1}": vals[i] for i in range(6)}
        return None

    # -------------------------
    # Crystal axes
    # -------------------------
    @cached_property
    def crystal_axes(self) -> list[list[float]] | None:
        """
        Returns:
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
        """
        m = re.search(
            r"crystal axes:.*?\n\s*a\(1\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*a\(2\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*a\(3\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)",
            self.text,
            re.I | re.S,
        )
        if m:
            nums = [float(x) for x in m.groups()]
            return np.array([nums[0:3], nums[3:6], nums[6:9]], dtype=float)
        return None

    # -------------------------
    # Reciprocal axes
    # -------------------------
    @cached_property
    def reciprocal_axes(self) -> list[list[float]] | None:
        m = re.search(
            r"reciprocal axes:.*?\n\s*b\(1\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*b\(2\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*b\(3\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)",
            self.text,
            re.I | re.S,
        )
        if m:
            nums = [float(x) for x in m.groups()]
            return np.array([nums[0:3], nums[3:6], nums[6:9]], dtype=float)
        return None

    # -------------------------
    # Pseudopotential details
    # -------------------------
    @cached_property
    def pseudopotentials(self) -> list[dict[str, Any]] | None:
        """
        Returns a list of dicts with pseudopotential details.
        """
        pseudos = []
        pattern = re.compile(
            r"PseudoPot\.\s*#\s*(\d+)\s*for\s+(\S+)\s+read from file:\s*(\S+)\s*"
            r"MD5 check sum:\s*([a-f0-9]+)\s*"
            r"Pseudo is\s*(.+?),\s*Zval\s*=\s*([\d.]+)\s*"
            r"Generated using\s*(.+?)\s*"
            r"Shape of augmentation charge:\s*(\S+)\s*"
            r"Using radial grid of\s*(\d+)\s*points,\s*(\d+)\s*beta functions.*?:\s*"
            r"((?:\s*l\(\d+\)\s*=\s*\d+\s*)+)"
            r"Q\(r\)\s*pseudized",
            re.I | re.S,
        )
        for m in pattern.finditer(self.text):
            l_block = m.group(11)
            l_values = [int(x) for x in re.findall(r"l\(\d+\)\s*=\s*(\d+)", l_block)]
            pseudos.append(
                {
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "file": m.group(3),
                    "md5": m.group(4),
                    "description": m.group(5).strip(),
                    "z_val": float(m.group(6)),
                    "generated_using": m.group(7).strip(),
                    "augmentation_shape": m.group(8),
                    "radial_grid_points": int(m.group(9)),
                    "n_beta": int(m.group(10)),
                    "l_values": l_values,
                }
            )
        return pseudos or None

    # -------------------------
    # Atomic species
    # -------------------------
    @cached_property
    def atomic_species(self) -> list[dict[str, Any]] | None:
        """
        Parses the 'atomic species' table.
        Returns:
            [
                {'symbol': 'Sr', 'valence': 10.0, 'mass': 87.62, 'pseudo': 'Sr( 1.00)'},
                {'symbol': 'V', 'valence': 13.0, 'mass': 50.9415, 'pseudo': 'V ( 1.00)'},
                {'symbol': 'O', 'valence': 6.0, 'mass': 15.9994, 'pseudo': 'O ( 1.00)'}
            ]
        """
        species = []
        m = re.search(
            r"atomic species\s+valence\s+mass\s+pseudopotential\n(.*?)\n\s*\n",
            self.text,
            re.S | re.I,
        )
        if m:
            for line in m.group(1).strip().splitlines():
                match = re.match(
                    r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+(.+?)\s*$", line
                )
                if match:
                    species.append(
                        {
                            "symbol": match.group(1),
                            "valence": float(match.group(2)),
                            "mass": float(match.group(3)),
                            "pseudo": match.group(4),
                        }
                    )
        return species or None

    # -------------------------
    # Atomic positions
    # -------------------------
    @cached_property
    def atomic_sites(self) -> list[dict[str, Any]]:
        """
        Parses the atomic positions block.
        Returns:
            [
                {'site': 1, 'symbol': 'Sr', 'tau_index': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                ...
            ]
        """
        atoms = []
        # Match lines like:
        #  1           Sr  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
        pattern = re.compile(
            r"^\s*(\d+)\s+(\S+)\s+tau\(\s*(\d+)\s*\)\s*=\s*\(\s*([-\d.Ee]+)\s+([-\d.Ee]+)\s+([-\d.Ee]+)\s*\)",
            re.M,
        )
        for m in pattern.finditer(self.text):
            atoms.append(
                {
                    "site": int(m.group(1)),
                    "symbol": m.group(2),
                    "tau_index": int(m.group(3)),
                    "x": float(m.group(4)),
                    "y": float(m.group(5)),
                    "z": float(m.group(6)),
                }
            )
        return atoms or None
    
    @cached_property
    def n_atoms(self) -> int | None:
        return self._system_info["nat"]
    
    @cached_property
    def atomic_positions(self) -> np.ndarray | None:
        atomic_position = np.zeros((self.n_atoms, 3), dtype=float)
        
        for atom in self.atomic_sites:
            atomic_position[atom["site"] - 1] = [atom["x"], atom["y"], atom["z"]]
        return atomic_position
    
    @cached_property
    def atomic_species(self) -> list[str]:
        atomic_species = []
        for atom in self.atomic_sites:
            atomic_species.append(atom["symbol"])
        return atomic_species
    
    @cached_property
    def _kpoints_fft_mem_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "n_kpoints": None,
            "smearing_type": None,
            "smearing_width_ry": None,
            "dense_gvecs": None,
            "dense_fft_dims": None,
            "smooth_gvecs": None,
            "smooth_fft_dims": None,
            "max_ram_per_proc_mb": None,
            "total_ram_mb": None,
            "negative_core_charge": None,
            "starting_charge": None,
            "renormalized_charge": None,
            "n_starting_wfcs": None,
            "cpu_time_so_far_s": None,
        }
        text = self.text

        # number of k points and smearing
        m = re.search(
            r"number of k points=\s*(\d+)\s+(\w+)\s+smearing,\s*width\s*\(Ry\)=\s*([\d.]+)",
            text,
            re.I,
        )
        if m:
            info["n_kpoints"] = int(m.group(1))
            info["smearing_type"] = m.group(2)
            info["smearing_width_ry"] = float(m.group(3))

        # dense grid
        m = re.search(
            r"Dense\s+grid:\s*(\d+)\s+G-vectors\s+FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)",
            text,
            re.I,
        )
        if m:
            info["dense_gvecs"] = int(m.group(1))
            info["dense_fft_dims"] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

        # smooth grid
        m = re.search(
            r"Smooth\s+grid:\s*(\d+)\s+G-vectors\s+FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)",
            text,
            re.I,
        )
        if m:
            info["smooth_gvecs"] = int(m.group(1))
            info["smooth_fft_dims"] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

        # RAM estimates
        m = re.search(
            r"Estimated max dynamical RAM per process >\s*([\d.]+)\s*MB",
            text,
            re.I,
        )
        if m:
            info["max_ram_per_proc_mb"] = float(m.group(1))

        m = re.search(
            r"Estimated total dynamical RAM >\s*([\d.]+)\s*MB",
            text,
            re.I,
        )
        if m:
            info["total_ram_mb"] = float(m.group(1))

        # negative core charge
        m = re.search(
            r"negative core charge=\s*([-\d.Ee]+)",
            text,
            re.I,
        )
        if m:
            info["negative_core_charge"] = float(m.group(1))

        # starting charge and renormalized
        m = re.search(
            r"starting charge\s*([\d.Ee+-]+),\s*renormalised to\s*([\d.Ee+-]+)",
            text,
            re.I,
        )
        if m:
            info["starting_charge"] = float(m.group(1))
            info["renormalized_charge"] = float(m.group(2))

        # starting wfcs
        m = re.search(
            r"Starting wfcs are\s*(\d+)",
            text,
            re.I,
        )
        if m:
            info["n_starting_wfcs"] = int(m.group(1))

        # total cpu time so far
        m = re.search(
            r"total cpu time spent up to now is\s*([\d.]+)\s*secs",
            text,
            re.I,
        )
        if m:
            info["cpu_time_so_far_s"] = float(m.group(1))

        return info

    # Public accessors
    @cached_property
    def n_kpoints(self) -> int | None:
        return self._kpoints_fft_mem_info["n_kpoints"]

    @cached_property
    def smearing_type(self) -> str | None:
        return self._kpoints_fft_mem_info["smearing_type"]

    @cached_property
    def smearing_width_ry(self) -> float | None:
        return self._kpoints_fft_mem_info["smearing_width_ry"]

    @cached_property
    def dense_gvecs(self) -> int | None:
        return self._kpoints_fft_mem_info["dense_gvecs"]

    @cached_property
    def dense_fft_dims(self) -> list[int] | None:
        return self._kpoints_fft_mem_info["dense_fft_dims"]

    @cached_property
    def smooth_gvecs(self) -> int | None:
        return self._kpoints_fft_mem_info["smooth_gvecs"]

    @cached_property
    def smooth_fft_dims(self) -> list[int] | None:
        return self._kpoints_fft_mem_info["smooth_fft_dims"]

    @cached_property
    def max_ram_per_proc_mb(self) -> float | None:
        return self._kpoints_fft_mem_info["max_ram_per_proc_mb"]

    @cached_property
    def total_ram_mb(self) -> float | None:
        return self._kpoints_fft_mem_info["total_ram_mb"]

    @cached_property
    def negative_core_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["negative_core_charge"]

    @cached_property
    def starting_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["starting_charge"]

    @cached_property
    def renormalized_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["renormalized_charge"]

    @cached_property
    def n_starting_wfcs(self) -> int | None:
        return self._kpoints_fft_mem_info["n_starting_wfcs"]

    @cached_property
    def cpu_time_so_far_s(self) -> float | None:
        return self._kpoints_fft_mem_info["cpu_time_so_far_s"]

    @cached_property
    def band_structure_info(self) -> dict[str, Any] | None:
        """
        Parses the 'Band Structure Calculation' section.
        Returns:
            {
                'method': str,
                'ethr': float,
                'avg_iterations': float,
                'cpu_time_so_far_s': float
            }
        """
        text = self.text

        # Match the block between "Band Structure Calculation" and "End of band structure calculation"
        m = re.search(
            r"Band Structure Calculation\s*(.*?)End of band structure calculation",
            text,
            re.I | re.S,
        )
        if not m:
            return None

        block = m.group(1)

        info: dict[str, Any] = {
            "method": None,
            "ethr": None,
            "avg_iterations": None,
            "cpu_time_so_far_s": None,
        }

        # Method line
        m2 = re.search(r"^\s*(.+diagonalization.+)$", block, re.I | re.M)
        if m2:
            info["method"] = m2.group(1).strip()

        # ethr and avg iterations
        m2 = re.search(
            r"ethr\s*=\s*([\d.Ee+-]+),\s*avg\s+#\s*of\s*iterations\s*=\s*([\d.]+)",
            block,
            re.I,
        )
        if m2:
            info["ethr"] = float(m2.group(1))
            info["avg_iterations"] = float(m2.group(2))

        # CPU time so far
        m2 = re.search(
            r"total cpu time spent up to now is\s*([\d.]+)\s*secs",
            block,
            re.I,
        )
        if m2:
            info["cpu_time_so_far_s"] = float(m2.group(1))

        return info
    
    @cached_property
    def is_scf_calculation(self) -> bool:
        return self.scf_iterations is not None
    
    @cached_property
    def is_bands_calculation(self) -> bool:
        return self.band_structure_info is not None

    @cached_property
    def scf_iterations(self) -> list[dict[str, Any]] | None:
        """
        Parses the 'Self-consistent Calculation' section into a list of dicts.
        Each dict contains:
            iteration, ecut, beta, ethr, avg_iterations,
            negative_rho_up, negative_rho_down,
            cpu_time_so_far_s, total_energy, estimated_scf_accuracy
        """
        iterations = []

        # Regex to match each iteration block
        pattern = re.compile(
            r"iteration\s*#\s*(\d+)\s+ecut=\s*([\d.]+)\s*Ry\s+beta=\s*([\d.]+).*?"
            r"ethr\s*=\s*([\d.Ee+-]+),\s*avg\s+#\s*of\s*iterations\s*=\s*([\d.]+).*?"
            r"(?:negative rho\s*\(up,\s*down\):\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s*)?"
            r"total cpu time spent up to now is\s*([\d.]+)\s*secs.*?"
            r"total energy\s*=\s*([-\d.]+)\s*Ry\s*"
            r"estimated scf accuracy\s*<\s*([\d.Ee+-]+)\s*Ry",
            re.I | re.S,
        )

        for m in pattern.finditer(self.text):
            iterations.append(
                {
                    "iteration": int(m.group(1)),
                    "ecut": float(m.group(2)),
                    "beta": float(m.group(3)),
                    "ethr": float(m.group(4)),
                    "avg_iterations": float(m.group(5)),
                    "negative_rho_up": float(m.group(6)) if m.group(6) else None,
                    "negative_rho_down": float(m.group(7)) if m.group(7) else None,
                    "cpu_time_so_far_s": float(m.group(8)),
                    "total_energy": float(m.group(9)),
                    "estimated_scf_accuracy": float(m.group(10)),
                }
            )

        return iterations or None
    
    @cached_property
    def final_results(self) -> dict[str, Any] | None:
        """
        Parses the final results section of the QE SCF output.
        Returns:
            {
                'fermi_energy_ev': float,
                'total_energy_final_ry': float,
                'total_all_electron_energy_ry': float,
                'estimated_scf_accuracy_ry': float,
                'smearing_contrib_ry': float,
                'internal_energy_ry': float,
                'energy_terms': {
                    'one_electron': float,
                    'hartree': float,
                    'xc': float,
                    'ewald': float,
                    'one_center_paw': float
                },
                'n_iterations_to_converge': int
            }
        """
        info: dict[str, Any] = {
            "fermi_energy_ev": None,
            "total_energy_final_ry": None,
            "total_all_electron_energy_ry": None,
            "estimated_scf_accuracy_ry": None,
            "smearing_contrib_ry": None,
            "internal_energy_ry": None,
            "energy_terms": {},
            "n_iterations_to_converge": None,
        }
        text = self.text

        # Fermi energy
        m = re.search(
            r"the Fermi energy is\s*([-\d.]+)\s*ev", text, re.I
        )
        if m:
            info["fermi_energy_ev"] = float(m.group(1))

        # Final total energy (with !)
        m = re.search(
            r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry", text, re.I
        )
        if m:
            info["total_energy_final_ry"] = float(m.group(1))

        # Total all-electron energy
        m = re.search(
            r"total all-electron energy\s*=\s*([-\d.]+)\s*Ry", text, re.I
        )
        if m:
            info["total_all_electron_energy_ry"] = float(m.group(1))

        # Estimated SCF accuracy
        m = re.search(
            r"estimated scf accuracy\s*<\s*([-\d.Ee+]+)\s*Ry", text, re.I
        )
        if m:
            info["estimated_scf_accuracy_ry"] = float(m.group(1))

        # Smearing contribution (-TS)
        m = re.search(
            r"smearing contrib\.\s*\(-TS\)\s*=\s*([-\d.]+)\s*Ry", text, re.I
        )
        if m:
            info["smearing_contrib_ry"] = float(m.group(1))

        # Internal energy
        m = re.search(
            r"internal energy E=F\+TS\s*=\s*([-\d.]+)\s*Ry", text, re.I
        )
        if m:
            info["internal_energy_ry"] = float(m.group(1))

        # Energy terms
        terms_patterns = {
            "one_electron": r"one-electron contribution\s*=\s*([-\d.]+)\s*Ry",
            "hartree": r"hartree contribution\s*=\s*([-\d.]+)\s*Ry",
            "xc": r"xc contribution\s*=\s*([-\d.]+)\s*Ry",
            "ewald": r"ewald contribution\s*=\s*([-\d.]+)\s*Ry",
            "one_center_paw": r"one-center paw contrib\.\s*=\s*([-\d.]+)\s*Ry",
        }
        for key, pat in terms_patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                info["energy_terms"][key] = float(m.group(1))

        # Number of iterations to converge
        m = re.search(
            r"convergence has been achieved in\s*(\d+)\s*iterations",
            text,
            re.I,
        )
        if m:
            info["n_iterations_to_converge"] = int(m.group(1))

        return info
    

    @cached_property
    def fermi_energy_ev(self) -> float | None:
        return self.final_results["fermi_energy_ev"]
    
    @cached_property
    def total_energy_final_ry(self) -> float | None:
        return self.final_results["total_energy_final_ry"]
    
    @cached_property
    def total_all_electron_energy_ry(self) -> float | None:
        return self.final_results["total_all_electron_energy_ry"]
    
    @cached_property
    def estimated_scf_accuracy_ry(self) -> float | None:
        return self.final_results["estimated_scf_accuracy_ry"]
    
    @cached_property
    def smearing_contrib_ry(self) -> float | None:
        return self.final_results["smearing_contrib_ry"]
    
    @cached_property
    def internal_energy_ry(self) -> float | None:
        return self.final_results["internal_energy_ry"]
    
    @cached_property
    def energy_terms(self) -> dict[str, float]:
        return self.final_results["energy_terms"]
    
    @cached_property
    def n_iterations_to_converge(self) -> int | None:
        return self.final_results["n_iterations_to_converge"]
    
    @cached_property
    def timing_info(self) -> dict[str, Any] | None:
        """
        Parses the timing breakdown section of QE output.
        Returns:
            {
                "output_data_dir": str,
                "timings": [
                    {"name": str, "cpu_s": float, "wall_s": float, "calls": int, "section": str|None},
                    ...
                ],
                "total_pwscf_cpu_s": float,
                "total_pwscf_wall_s": float,
                "terminated_on": str
            }
        """
        text = self.text
        info: dict[str, Any] = {
            "output_data_dir": None,
            "timings": [],
            "total_pwscf_cpu_s": None,
            "total_pwscf_wall_s": None,
            "terminated_on": None,
        }

        # Output data dir
        m = re.search(
            r"Writing all to output data dir\s+(\S+)", text, re.I
        )
        if m:
            info["output_data_dir"] = m.group(1)

        # Regex for timing lines
        timing_line = re.compile(
            r"^\s*([A-Za-z0-9_:*]+)\s*:\s*([\d.]+)s CPU\s+([\d.]+)s WALL\s*\(\s*(\d+)\s*calls?\)",
            re.M,
        )

        current_section = None
        for line in text.splitlines():
            # Section headers
            sec_match = re.match(r"^\s*Called by\s+(.+?):", line, re.I)
            if sec_match:
                current_section = sec_match.group(1).strip()
                continue
            if re.match(r"^\s*General routines", line, re.I):
                current_section = "General routines"
                continue
            if re.match(r"^\s*Parallel routines", line, re.I):
                current_section = "Parallel routines"
                continue

            # Timing lines
            m = timing_line.match(line)
            if m:
                info["timings"].append(
                    {
                        "name": m.group(1),
                        "cpu_s": float(m.group(2)),
                        "wall_s": float(m.group(3)),
                        "calls": int(m.group(4)),
                        "section": current_section,
                    }
                )

        # Total PWSCF time
        m = re.search(
            r"PWSCF\s*:\s*([\d.]+)s CPU\s+([\d.]+)s WALL", text, re.I
        )
        if m:
            info["total_pwscf_cpu_s"] = float(m.group(1))
            info["total_pwscf_wall_s"] = float(m.group(2))

        # Termination time
        m = re.search(
            r"This run was terminated on:\s*(.+)", text, re.I
        )
        if m:
            info["terminated_on"] = m.group(1).strip()

        return info
    
    @cached_property
    def output_data_dir(self) -> str | None:
        return self.timing_info["output_data_dir"]
    
    @cached_property
    def timings(self) -> dict[str, Any] | None:
        return self.timing_info["timings"]
    
    @cached_property
    def total_pwscf_cpu_s(self) -> float | None:
        return self.timing_info["total_pwscf_cpu_s"]
    
    @cached_property
    def total_pwscf_wall_s(self) -> float | None:
        return self.timing_info["total_pwscf_wall_s"]
    
    @cached_property
    def terminated_on(self) -> str | None:
        return self.timing_info["terminated_on"]
    
    
    
    
class PwXML:
    """Parser for the XML output of the PW module in Quantum ESPRESSO."""
    
    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._tree = ET.parse(self._filepath)
        self._root = self._tree.getroot()
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @cached_property
    def root(self) -> ET.Element:
        return self._root
    
    @cached_property
    def tree(self) -> ET.ElementTree:
        return self._tree
    
    @cached_property
    def is_non_colinear(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/noncolin")
        if match:
            return str2bool(match[0].text)
        return None
    
    @cached_property
    def is_spin_calc(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/lsda")
        if match:
            return str2bool(match[0].text)
        return None
    
    @cached_property
    def is_spin_orbit_calc(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/spinorbit")
        if match:
            return str2bool(match[0].text)
        return None
    
    def _parse_magnetization(self, main_xml_root):
        """A helper method to parse the magnetization tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        
    @cached_property
    def spin_orbit_orbitals(self) -> list[dict]:
        return [
                {"l": "s", "j": 0.5, "m": -0.5},
                {"l": "s", "j": 0.5, "m": 0.5},
                {"l": "p", "j": 0.5, "m": -0.5},
                {"l": "p", "j": 0.5, "m": 0.5},
                {"l": "p", "j": 1.5, "m": -1.5},
                {"l": "p", "j": 1.5, "m": -0.5},
                {"l": "p", "j": 1.5, "m": -0.5},
                {"l": "p", "j": 1.5, "m": 1.5},
                {"l": "d", "j": 1.5, "m": -1.5},
                {"l": "d", "j": 1.5, "m": -0.5},
                {"l": "d", "j": 1.5, "m": -0.5},
                {"l": "d", "j": 1.5, "m": 1.5},
                {"l": "d", "j": 2.5, "m": -2.5},
                {"l": "d", "j": 2.5, "m": -1.5},
                {"l": "d", "j": 2.5, "m": -0.5},
                {"l": "d", "j": 2.5, "m": 0.5},
                {"l": "d", "j": 2.5, "m": 1.5},
                {"l": "d", "j": 2.5, "m": 2.5},
            ]
        return orbitals
    
    @cached_property
    def colinear_orbitals(self) -> list[dict]:
        orbitals = [
                {"l": 0, "m": 1},
                {"l": 1, "m": 3},
                {"l": 1, "m": 1},
                {"l": 1, "m": 2},
                {"l": 2, "m": 5},
                {"l": 2, "m": 3},
                {"l": 2, "m": 1},
                {"l": 2, "m": 2},
                {"l": 2, "m": 4},
            ]
        return orbitals
    
    @cached_property
    def colinear_orbital_names(self) -> list[str]:
        return [
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
    
    @cached_property
    def orbitals(self) -> list[dict]:
        if self.is_non_colinear:
            return self.spin_orbit_orbitals
        else:
            return self.colinear_orbitals
    
    @cached_property
    def orbital_names(self) -> list[str]:
        orbital_names = []
        if self.is_non_colinear:
            for orbital in self.spin_orbit_orbitals:
                tmp_name = ""
                for key, value in orbital.items():
                    # print(key,value)
                    if key != "l":
                        tmp_name = tmp_name + key + str(value)
                    else:
                        tmp_name = tmp_name + str(value) + "_"
                orbital_names.append(tmp_name)
            return orbital_names
        else:
            return self.colinear_orbital_names
        
    @cached_property
    def n_orbitals(self) -> int:
        return len(self.orbitals)
    
    @cached_property
    def n_spin(self) -> int:
        if self.is_non_colinear:
            return 4
        elif self.is_spin_calc:
            return 2
        else:
            return 1
 
    @cached_property
    def atm_wfc(self) -> int:
        match = self.root.findall(".//output/band_structure/num_of_atomic_wfc")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def n_electrons(self) -> float | None:
        match = self.root.findall(".//output/band_structure/nelec")
        if match:
            return float(match[0].text)
        return None

    @cached_property
    def n_bands(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_up")
        if match:
            return int(match[0].text)
        
        match = self.root.findall(".//output/band_structure/nbnd")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def n_bands_up(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_up")
        if match:
            return int(match[0].text)
        return 0

    @cached_property
    def n_bands_down(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_down")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def n_kpoints(self) -> int:
        match = self.root.findall(".//output/band_structure/nks")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def reciprocal_lattice(self) -> np.ndarray | None:
        match = self.root.findall(".//output/basis_set/reciprocal_lattice")
        if match:
            lattice_vectors = []
            for acell in match[0]:
                lattice_vectors.append(np.array(acell.text.split(), dtype=float))
            return np.array(lattice_vectors, dtype=float)
        return None
    
    @cached_property
    def direct_lattice(self) -> np.ndarray | None:
        match = self.root.findall(".//output/atomic_structure/cell")
        if match:
            lattice_vectors = []
            for acell in match[0]:
                lattice_vectors.append(np.array(acell.text.split(), dtype=float))
            return np.array(lattice_vectors, dtype=float) * AU_TO_ANG
        return None
    
    @cached_property
    def atomic_sites(self) -> np.ndarray | None:
        match = self.root.findall(".//output/atomic_structure/atomic_positions")
        if match:
            atomic_sites = {"species": [], "positions": [], "index": []}
            for ion in match[0]:
                atomic_sites["species"].append(ion.attrib["name"])
                atomic_sites["positions"].append(ion.text.split())
                atomic_sites["index"].append(int(ion.attrib["index"]))
            
            atomic_sites["positions"] = np.array(atomic_sites["positions"], dtype=float)
            return atomic_sites
        return None
    
    @cached_property
    def atomic_positions(self) -> np.ndarray | None:
        if self.atomic_sites:
            return self.atomic_sites["positions"]
        return None
    
    @cached_property
    def atomic_species(self) -> list[str] | None:
        if self.atomic_sites:
            return self.atomic_sites["species"]
        return None
    
    @cached_property
    def atomic_indices(self) -> list[int] | None:
        if self.atomic_sites:
            return self.atomic_sites["index"]
        return None
    
    @cached_property
    def specie_types(self) -> dict[str, float] | None:
        match = self.root.findall(".//output/atomic_structure/atomic_species")
        if match:
            atomic_sites = {"species": [], 
                            "mass": [], 
                            "pseudo_file": [], 
                            "starting_magnetization": [],
                            "spin_teta": []}
            for ion in match[0]:
                atomic_sites["species"].append(ion.attrib["name"])
                atomic_sites["mass"].append(float(ion.attrib["mass"]))
                atomic_sites["pseudo_file"].append(ion.attrib["pseudo_file"])
                atomic_sites["starting_magnetization"].append(float(ion.attrib["starting_magnetization"]))
                atomic_sites["spin_teta"].append(float(ion.attrib["spin_teta"]))
            
            atomic_sites["mass"] = np.array(atomic_sites["mass"], dtype=float)
            atomic_sites["starting_magnetization"] = np.array(atomic_sites["starting_magnetization"], dtype=float)
            atomic_sites["spin_teta"] = np.array(atomic_sites["spin_teta"], dtype=float)
            return atomic_sites
        return None
    
    @cached_property
    def n_species_types(self) -> int:
        if self.specie_types:
            return len(self.specie_types["species"])
        return 0
    
    @cached_property
    def compositions(self) -> dict[str, int]:
        if self.atomic_species:
            species_types = self.specie_types["species"]
            compositions = {specie_name: 0 for specie_name in species_types}
            
            for specie_name in self.atomic_species:
                compositions[specie_name] += 1
            return compositions
        return {}
    
    @cached_property
    def n_atoms(self) -> int:
        if self.atomic_species:
            return len(self.atomic_species)
        return 0
    
        
    @cached_property
    def alat(self) -> float | None:
        match = self.root.findall(".//output/atomic_structure")
        if match:
            return float(match[0].attrib["alat"]) * AU_TO_ANG
        return None
    
    @cached_property
    def ks_energies(self) -> None:
        ks_energies_match = self.root.findall(".//output/band_structure/ks_energies")
        if not ks_energies_match:
            return None
        
        logger.debug(f"n_ks_energies: {len(ks_energies_match)}")

        raw_n_kpoints = self.n_kpoints if self.is_spin_calc else self.n_kpoints * 2
        raw_n_bands = self.n_bands if not self.is_spin_calc else self.n_bands * 2


        raw_bands = np.zeros(shape=(self.n_kpoints, raw_n_bands))
        raw_occupations = np.zeros(shape=(self.n_kpoints, raw_n_bands))
        kpoints = np.zeros(shape=(self.n_kpoints, 3))
        weights = np.zeros(shape=(self.n_kpoints))
        bands = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin))
        occupations = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin))
        npws = np.zeros(shape=(self.n_kpoints, self.n_bands))

        
        for ikpoint, kpoint_element in enumerate(ks_energies_match):

            kpoints_match = kpoint_element.findall(".//k_point")
            if kpoints_match:
                kpoints[ikpoint, :] = np.array(kpoints_match[0].text.split(), dtype=float)
            
            weight_match = kpoint_element.findall(".//k_point")
            if weight_match:
                weights[ikpoint] = np.array(weight_match[0].attrib["weight"], dtype=float)
            
            eigenvalues_match = kpoint_element.findall(".//eigenvalues")
            if eigenvalues_match:
                raw_bands[ikpoint, :] = np.array(eigenvalues_match[0].text.split(), dtype=float)
            
            occupations_match = kpoint_element.findall(".//occupations")
            if occupations_match:
                raw_occupations[ikpoint, :] = np.array(occupations_match[0].text.split(), dtype=float)

            npws_match = kpoint_element.findall(".//npw")
            if npws_match:
                npws[ikpoint, :] = np.array(npws_match[0].text.split(), dtype=int)

            if self.is_spin_calc:
                bands[ikpoint, :, 0] = raw_bands[ikpoint, :self.n_bands_up]
                bands[ikpoint, :, 1] = raw_bands[ikpoint, self.n_bands_up:]
                occupations[ikpoint, :, 0] = raw_occupations[ikpoint, :self.n_bands_up]
                occupations[ikpoint, :, 1] = raw_occupations[ikpoint, self.n_bands_up:]
            else:
                bands[ikpoint, :, 0] = raw_bands[ikpoint, :]
                occupations[ikpoint, :, 0] = raw_occupations[ikpoint, :]
                
        kpoints = kpoints * (2 * np.pi / self.alat)
        # Converting back to crystal basis
        kpoints = np.around(
            kpoints.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8
        )
        # print(bands[:,:,0].shape)
        # print(bands[:,self.n_bands:,1].shape)
        # print(np.allclose(bands[...,0], bands[...,1]))
        
        ks_energies = {"bands": bands, "occupations": occupations, "kpoints": kpoints, "weights": weights}
        return ks_energies
    
    
    @cached_property
    def kpoints(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["kpoints"]
        return None
    
    @cached_property
    def weights(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["weights"]
        return None
    
    @cached_property
    def bands(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["bands"]
        return None
    
    @cached_property
    def occupations(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["occupations"]
        return None
    
    @cached_property
    def symmetries_element(self) -> ET.Element | None:
        match = self.root.findall(".//output/symmetries")
        if match:
            return match[0]
        return None
    
    @cached_property
    def n_symmetries(self) -> int:
        match = self.symmetries_element.findall(".//nsym")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def n_rotations(self) -> int:
        match = self.symmetries_element.findall(".//nrot")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def spg(self) -> int:
        match = self.symmetries_element.findall(".//space_group")
        if match:
            return int(match[0].text)
        return 0
    
    @cached_property
    def symmetry_operations_elements(self) -> ET.Element | None:
        match = self.symmetries_element.findall(".//symmetries")
        if match:
            return match[0]
        return None
    
    @cached_property
    def n_sym_ops(self) -> int:
        n_sym_match = self.symmetries_element.findall(".//nsym")
        if n_sym_match:
            return int(n_sym_match[0].text)
        return 0
    
    @cached_property
    def n_rot(self) -> int:
        n_rot_match = self.symmetries_element.findall(".//nrot")
        if n_rot_match:
            return int(n_rot_match[0].text)
        return 0
    
    @cached_property
    def sym_ops(self) -> dict[str, Any] | None:
        sym_ops_match = self.symmetries_element.findall(".//symmetry")
        if sym_ops_match:
            sym_ops = {"rotations": [], "translations": [], "equivalent_atoms": []}
            
            for symmetry_operation in sym_ops_match:
                rotation = symmetry_operation.findall(".//rotation")
                if rotation:
                    rotation = np.array(rotation[0].text.split(), dtype=float)
                else:
                    rotation = np.eye(3)
                rotation = rotation.reshape(3, 3).T
                
                sym_ops["rotations"].append(rotation)
                
                fractional_translation = symmetry_operation.findall(".//fractional_translation")
                if fractional_translation:
                    sym_ops["translations"].append(np.array(fractional_translation[0].text.split(), dtype=float))
                else:
                    sym_ops["translations"].append(np.zeros(3))
                
                equivalent_atoms = symmetry_operation.findall(".//equivalent_atoms")
                if equivalent_atoms:
                    sym_ops["equivalent_atoms"].append(np.array(equivalent_atoms[0].text.split(), dtype=int))
                else:
                    sym_ops["equivalent_atoms"].append(np.zeros(5))
                
            sym_ops["rotations"] = np.array(sym_ops["rotations"], dtype=float)
            sym_ops["translations"] = np.array(sym_ops["translations"], dtype=float)
            return sym_ops
        return None
    
    @cached_property
    def rotations(self) -> np.ndarray | None:
        if self.sym_ops:
            return self.sym_ops["rotations"]
        return None
    
    @cached_property
    def fermi(self) -> float | None:
        match = self.root.findall(".//output/band_structure/fermi_energy")
        if match:
            return float(match[0].text) * HARTREE_TO_EV
        return None
    
 
        
    @cached_property
    def kmesh_mode(self) -> str:
        monkhorst_pack_match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        gamma_point_match = self.root.findall(".//output/band_structure/starting_k_points/gamma_point")
        if monkhorst_pack_match:
            return "monkhorst_pack"
        elif gamma_point_match:
            return "gamma_point"
        return 0
        
    @cached_property
    def nk1(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk1"])
        return 0
    
    @cached_property
    def nk2(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk2"])
        return 0
    
    @cached_property
    def nk3(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk3"])
        return 0
    
    @cached_property
    def sk1(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k1"])
        return 0
    
    @cached_property
    def sk2(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k2"])
        return 0
    
    @cached_property
    def sk3(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k3"])
        return 0