import logging
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
ROOT_TEST_DIR = CURRENT_DIR.parent
ROOT_DIR = ROOT_TEST_DIR.parent

sys.path.append(str(ROOT_DIR))

from pyprocar import io
from tests.utils.data import ALL_TEST_CASES, DOS_CALC_TYPES, EBS_CALC_TYPES

user_logger = logging.getLogger("user")

for calc_test_case in ALL_TEST_CASES:
    path = calc_test_case.path

    mat_system = calc_test_case.mat_system
    code = calc_test_case.code
    version = calc_test_case.version
    mag_type = calc_test_case.mag_type
    calc_type = calc_test_case.calc_type

    parser = io.Parser(
        code=calc_test_case.code,
        dirpath=calc_test_case.path,
    )

    if calc_type in DOS_CALC_TYPES:
        if parser.dos:
            parser.dos.save(path / "dos.pkl")
        else:
            user_logger.warning(f"No DOS found for {calc_test_case.get_id()}")
    elif calc_type in EBS_CALC_TYPES:
        if parser.ebs:
            parser.ebs.save(path / "ebs.pkl")
        else:
            user_logger.warning(f"No EBS found for {calc_test_case.get_id()}")

    if parser.structure:
        parser.structure.save(path / "structure.pkl")
    else:
        user_logger.warning(f"No structure found for {calc_test_case.get_id()}")
