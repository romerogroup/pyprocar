import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

QEScalar = Union[bool, int, float, str]
QETensor = Dict[Tuple[int, ...], QEScalar]
QEValue = Union[QEScalar, QETensor]


    
def _strip_qe_comments(text: str) -> str:
    """Remove QE-style comments from text.

    - In namelists, "!" starts a comment anywhere on the line (unless inside quotes)
    - In cards, lines with first non-space character "#" are comments
    - Trailing whitespace-only lines are dropped
    """
    cleaned_lines = []
    for line in text.splitlines():
        # Skip card-style comment lines beginning with '#'
        if line.lstrip().startswith("#"):
            continue

        in_single = False
        in_double = False
        out_chars: list[str] = []
        for ch in line:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double

            # In namelists, '!' begins a comment when not inside quotes
            if ch == "!" and not in_single and not in_double:
                break
            out_chars.append(ch)

        cleaned = "".join(out_chars).rstrip()
        if cleaned.strip():
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def parse_qe_input_cards(text: str | list[str]) -> Dict[str, QEValue]:
    """Parse the input cards in Quantum ESPRESSO.
    Input data format: { } = optional, [ ] = it depends, | = or

    All quantities whose dimensions are not explicitly specified are in
    RYDBERG ATOMIC UNITS. Charge is "number" charge (i.e. not multiplied
    by e); potentials are in energy units (i.e. they are multiplied by e).

    BEWARE: TABS, CRLF, ANY OTHER STRANGE CHARACTER, ARE A SOURCES OF TROUBLE
    USE ONLY PLAIN ASCII TEXT FILES (CHECK THE FILE TYPE WITH UNIX COMMAND "file")

    Namelists must appear in the order given below.
    Comment lines in namelists can be introduced by a "!", exactly as in
    fortran code. Comments lines in cards can be introduced by
    either a "!" or a "#" character in the first position of a line.
    Do not start any line in cards with a "/" character.
    Leave a space between card names and card options, e.g.
    ATOMIC_POSITIONS (bohr), not ATOMIC_POSITIONS(bohr)
    
    """
    params: Dict[str, QEValue] = {}
    if isinstance(text, list):
        text = "\n".join(text)

    # Remove comments before tokenizing
    text = _strip_qe_comments(text)

    token_regex = re.compile(r"([A-Za-z0-9_]+(?:\s*\([^)]*\))?)\s*=\s*([^,\n]+)")
    for name, value_str in token_regex.findall(text):
        name = name.strip()
        base_name, index_tuple = split_name_indices(name)
        typed_val = convert_to_typed_value(value_str.strip())
        key = base_name.lower()
        if index_tuple is None:
            params[key] = typed_val
        else:
            existing = params.get(key)
            if not isinstance(existing, dict):
                existing = {}
                params[key] = existing  # type: ignore[assignment]
            assert isinstance(existing, dict)
            existing[index_tuple] = typed_val
    return params


def split_name_indices(name: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
    m = re.match(r"^([A-Za-z0-9_]+)\s*\(([^)]*)\)$", name)
    if not m:
        return name, None
    base = m.group(1)
    indices_raw = m.group(2).strip()
    if not indices_raw:
        return base, tuple()
    parts = [p.strip() for p in indices_raw.split(",")]
    idx: List[int] = []
    for p in parts:
        try:
            idx.append(int(p))
        except ValueError:
            pass
    return base, tuple(idx)

def convert_to_typed_value(raw: str) -> QEScalar:
    s = raw.strip().rstrip(",")
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    low = s.lower()
    if low in (".true.", ".false."):
        return low == ".true."
    s2 = s.replace("D", "E").replace("d", "e")
    if re.fullmatch(r"[+-]?\d+", s2):
        try:
            return int(s2)
        except Exception:
            pass
    try:
        return float(s2)
    except Exception:
        return s


# ===== Base Class =====
@dataclass
class QECardBlock:
    name: str
    options: str
    block: str