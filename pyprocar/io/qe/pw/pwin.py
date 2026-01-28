__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path

import numpy as np

from pyprocar.io.qe.utils import (
    QECardBlock,
    QEValue,
    parse_qe_input_cards,
)

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


# ===== Specialized Subclasses =====
@dataclass
class ControlCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class SystemCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class ElectronsCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class IonsCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class CellCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class FCPCard(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class RISM(QECardBlock):
    _data: dict[str, QEValue] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        self._data = parse_qe_input_cards(self.block)

    @property
    def data(self) -> dict[str, QEValue]:
        return self._data


@dataclass
class AtomicSpeciesCard(QECardBlock):
    labels: list[str] = field(default_factory=list)
    masses: np.ndarray | None = None
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
                m = float(mass_str.replace("D", "E").replace("d", "e"))
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
    def species(self) -> dict[str, float]:
        mapping: dict[str, float] = {}
        if self.labels and self.masses is not None:
            for lbl, m in zip(self.labels, self.masses.tolist()):
                mapping[lbl] = m
        return mapping


@dataclass
class AtomicPositionsCard(QECardBlock):
    # Parsed attributes
    mode: str | None = None  # one of: alat, bohr, angstrom, crystal, crystal_sg
    labels: list[str] = field(default_factory=list)
    positions: np.ndarray | None = None  # shape (nat, 3) when 3 coords provided
    constraints: np.ndarray | None = None  # shape (nat, 3) integers (0/1)
    wyckoff: list[str | None] = field(default_factory=list)  # for crystal_sg
    wyckoff_params: list[tuple[float | None, float | None, float | None]] = field(
        default_factory=list
    )

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
        if expr.startswith("+"):
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
            ast.Tuple,  # not expected but harmless if encountered
            ast.Load,
            ast.Mod,  # not used, but reject at validation below
        )

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("Invalid constant in expression")
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
                    return left**right
                raise ValueError("Unsupported binary operator")
            # Reject all other nodes (Names, Calls, etc.)
            raise ValueError("Unsupported expression element")

        tree = ast.parse(expr, mode="eval")
        for n in ast.walk(tree):
            if not isinstance(n, allowed_nodes):
                raise ValueError("Disallowed token in expression")
        return float(_eval(tree))

    @staticmethod
    def _parse_if_pos(raw: str) -> tuple[int, int, int]:
        tmp = raw.replace("{", " ").replace("}", " ").strip()
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
        mode_raw = mode_raw.replace("{", "").replace("}", "")
        mode_raw = mode_raw.replace("(", "").replace(")", "")
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
        wyckoff_list: list[str | None] = []
        wyckoff_params_list: list[tuple[float | None, float | None, float | None]] = []

        wyckoff_re = re.compile(r"^[0-9]+[A-Za-z]+$")

        for line in lines:
            # Split off optional constraints in braces
            if "{" in line and "}" in line:
                line_part, brace_part = line.split("{", 1)
                brace_content = "{" + brace_part
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
                    pvals: list[float | None] = []
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
    line_points: np.ndarray | None = None
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
    modified_knames: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parse()

    def parse(self) -> None:
        """Parse the K_POINTS card body according to its option/mode."""
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
        _ = int(lines[0])  # n_kpoints - not used but parsed for validation
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
        high_symmetry_points: list[list[float]] = []
        line_points: list[int] = []
        for line in lines[1:]:
            if line.strip():
                cols = line.split()
                kx, ky, kz, n_points = cols[:4]
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
            tmp_knames: list[str] = []
            for comment in self.line_comments:
                tmp_knames.append(
                    comment.replace(",", "").replace("vlvp1d", "").replace(" ", "")
                )
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

        self.kpoints = kpts_arr
        self.weights = wts_arr
        self.line_comments = line_comments


@dataclass
class OccupationsCard(QECardBlock):
    @cached_property
    def data(self) -> dict[str, QEValue]:
        return parse_qe_input_cards(self.block)


@dataclass
class ConstraintsCard(QECardBlock):
    @cached_property
    def data(self) -> dict[str, QEValue]:
        return parse_qe_input_cards(self.block)


@dataclass
class AdditionalKPointsCard(QECardBlock):
    @cached_property
    def data(self) -> dict[str, QEValue]:
        return parse_qe_input_cards(self.block)


@dataclass
class SolventsCard(QECardBlock):
    @cached_property
    def data(self) -> dict[str, QEValue]:
        return parse_qe_input_cards(self.block)


@dataclass
class HubbardCard(QECardBlock):
    @cached_property
    def data(self) -> dict[str, QEValue]:
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
    def keys(cls) -> list[str]:
        return list(cls.__members__.keys())

    @classmethod
    def is_in(cls, name: str) -> bool:
        return name.lower() in cls.__members__.keys()


# ===== Parsing / Extraction =====
def extract_qe_input_blocks(text: str) -> list[QECardBlock]:
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
    known_cards = [name for name in QECardBlockEnum.keys() if name not in ("unknown",)]
    known_headers = "|".join(re.escape(n.upper()) for n in known_cards)
    card_pattern = (
        rf"^(?P<card_name>(?:{known_headers}))"
        rf"(?:[ \t]+(?P<card_options>(?:\{{[^}}]*\}}|[^\n]+)))?"  # options only on same line
        rf"[ \t]*\n"
        rf"(?P<card_body>(?:(?!^(?:{known_headers})\b|^&).*(?:\n|$))*)"
    )

    combined_pattern = f"(?:{namelist_pattern})|(?:{card_pattern})"

    blocks: list[QECardBlock] = []
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
            raise ValueError(
                f"Error with parsing QE input blocks: Invalid card or namelist: {match.group()}"
            )
        if QECardBlockEnum.is_in(name):
            blocks.append(QECardBlockEnum.from_name(name).value(name, options, block))
        else:
            blocks.append(QECardBlockEnum["unknown"].value(name, options, block))

    return blocks


class PwIn:
    """Parser for the input of the PW module in Quantum ESPRESSO with structured sections (no dataclass)."""

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
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
                re.search(r"^.*&control\b", head, re.IGNORECASE | re.MULTILINE)
                is not None
                and re.search(r"^.*&system\b", head, re.IGNORECASE | re.MULTILINE)
                is not None
            )
        except Exception:
            return False

    _filepath: Path
    _text: str

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()

    def _read(self) -> str:
        with open(self.filepath) as f:
            return f.read()

    @cached_property
    def text(self) -> str:
        return self._text

    @property
    def filepath(self) -> Path:
        return self._filepath

    @cached_property
    def data(self) -> dict[str, QECardBlock]:
        qe_card_blocks = extract_qe_input_blocks(self.text)
        return {block.name: block for block in qe_card_blocks}

    @cached_property
    def control_card(self) -> ControlCard:
        for block in self.data.values():
            if isinstance(block, ControlCard):
                return block
        raise ValueError("ControlCard not found in PWInput")

    @cached_property
    def system_card(self) -> SystemCard:
        for block in self.data.values():
            if isinstance(block, SystemCard):
                return block
        raise ValueError("SystemCard not found in PWInput")

    @cached_property
    def electrons_card(self) -> ElectronsCard:
        for block in self.data.values():
            if isinstance(block, ElectronsCard):
                return block
        raise ValueError("ElectronsCard not found in PWInput")

    @cached_property
    def atomic_species_card(self) -> AtomicSpeciesCard:
        for block in self.data.values():
            if isinstance(block, AtomicSpeciesCard):
                return block
        raise ValueError("AtomicSpeciesCard not found in PWInput")

    @cached_property
    def atomic_positions_card(self) -> AtomicPositionsCard:
        for block in self.data.values():
            if isinstance(block, AtomicPositionsCard):
                return block
        raise ValueError("AtomicPositionsCard not found in PWInput")

    @cached_property
    def kpoints_card(self) -> KPointsCard:
        for card in self.data.values():
            if isinstance(card, KPointsCard):
                return card
        raise ValueError("KPointsCard not found in PWInput")

    @cached_property
    def cell_card(self) -> CellCard:
        for block in self.data.values():
            if isinstance(block, CellCard):
                return block
        raise ValueError("CellCard not found in PWInput")

    @cached_property
    def occupations_card(self) -> OccupationsCard:
        for block in self.data.values():
            if isinstance(block, OccupationsCard):
                return block
        raise ValueError("OccupationsCard not found in PWInput")

    @cached_property
    def bands_kpoint_names(self) -> list[str] | None:
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
        calc_mode: QEValue | None = None
        for b in blocks:
            if isinstance(b, ControlCard):
                data = b.data
                calc_mode = data.get("calculation") if isinstance(data, dict) else None
                if isinstance(calc_mode, str):
                    calc_mode = calc_mode.strip().lower().strip("'\"")
                break

        if calc_mode != "bands":
            return None

        # Find K_POINTS block
        kpoints_block: KPointsCard | None = None
        for b in blocks:
            if isinstance(b, KPointsCard):
                kpoints_block = b
                break
        if kpoints_block is None:
            return None

        mode = (
            (kpoints_block.options or "").strip().lower()
            if hasattr(kpoints_block, "options")
            else ""
        )
        if mode not in {
            "crystal",
            "crystal_b",
            "tpiba",
            "tpiba_b",
            "crystal_c",
            "tpiba_c",
            "gamma",
        }:
            # Unknown or unsupported for band labels
            return None

        raw = (kpoints_block.block or "") if hasattr(kpoints_block, "block") else ""
        if not raw:
            return None

        raw_lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not raw_lines:
            return None

        def _label_from_line(line: str) -> str | None:
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
            point_lines = raw_lines[1 : 1 + n_declared]
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
