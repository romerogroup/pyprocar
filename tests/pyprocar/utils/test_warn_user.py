import functools
import importlib
import linecache
import os
import subprocess
import sys
import threading
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest

import pyprocar.utils.log_utils as log_utils
from pyprocar.core.kpoints import KPath
from tests.pyprocar.scripts import test_fermi_handler_save
from tests.utils.user_warning import user_warning

ROOT = Path(__file__).resolve().parents[3]
KPOINTS = np.vstack(
    [np.linspace([0, 0, 0], [0.5, 0, 0], 5), np.linspace([0.5, 0.5, 0.5], [0.5, 0.5, 0], 5)]
)
NAMES = [("G", "X")]
SPECIAL = {"G": KPOINTS[0], "X": KPOINTS[4]}
MESSAGE = "KPath got 1 segment names for 2 segments in the k-points; ticks use 1"
CALL = "KPath(kpoints=KPOINTS, segment_names=NAMES, special_kpoint_map=SPECIAL)"
USER_CODE = f"x = 1\n{CALL}\n"
ONE_NAME_FOR_TWO_SEGMENTS = functools.partial(
    KPath, kpoints=KPOINTS, segment_names=NAMES, special_kpoint_map=SPECIAL
)
handler = test_fermi_handler_save.handler


def _next_line() -> int:
    return sys._getframe(1).f_lineno + 1


def _source(warning: warnings.WarningMessage) -> str:
    return linecache.getline(warning.filename, warning.lineno).strip()


def _record(call) -> list[tuple[str, str, int]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    return [(w.category.__name__, w.filename, w.lineno) for w in record]


def _namespace(**names: object) -> dict[str, object]:
    return {"KPath": KPath, "KPOINTS": KPOINTS, "NAMES": NAMES, "SPECIAL": SPECIAL, **names}


def _script_namespace(script: Path) -> dict[str, object]:
    return _namespace(
        __name__="__main__",
        __spec__=None,
        __file__=str(script),
        __loader__=SourceFileLoader("__main__", str(script)),
        __builtins__=__builtins__,
    )


def _labelled_kpath() -> KPath:
    return KPath(kpoints=KPOINTS, segment_names=NAMES, special_kpoint_map=SPECIAL)


def test_a_notebook_cell_gets_the_warning_at_its_own_line():
    notebook = vars(types.ModuleType("__main__"))
    notebook.update(_namespace())
    cell = compile(USER_CODE, "<ipython-input-1-abc>", "exec")

    recorded = _record(lambda: exec(cell, notebook))

    assert recorded == [("UserWarning", "<ipython-input-1-abc>", 2)]


def test_a_script_gets_the_warning_at_its_own_line_and_no_other_warning(tmp_path):
    script = tmp_path / "user.py"
    script.write_text(USER_CODE)
    code = compile(USER_CODE, str(script), "exec")

    recorded = _record(lambda: exec(code, _script_namespace(script)))

    assert recorded == [("UserWarning", str(script), 2)]


def test_an_error_filter_raises_the_user_warning_out_of_a_script(tmp_path):
    script = tmp_path / "user.py"
    script.write_text(USER_CODE)
    code = compile(USER_CODE, str(script), "exec")

    with warnings.catch_warnings(), pytest.raises(UserWarning, match=MESSAGE):
        warnings.simplefilter("error")
        exec(code, _script_namespace(script))


@pytest.mark.parametrize(
    ("error_filter", "returncode", "expected_line"),
    [
        ('"error"', 1, "UserWarning: {message}"),
        ('"error", DeprecationWarning', 0, "{script}:8: UserWarning: {message}"),
    ],
)
def test_a_python_script_sees_the_user_warning_under_an_error_filter(
    tmp_path, error_filter, returncode, expected_line
):
    script = tmp_path / "user.py"
    script.write_text(
        "\n".join(
            [
                "import warnings",
                "import numpy as np",
                "from pyprocar.core.kpoints import KPath",
                "k = np.vstack([np.linspace([0, 0, 0], [.5, 0, 0], 5),",
                "               np.linspace([.5, .5, .5], [.5, .5, 0], 5)])",
                "n, m = [('G', 'X')], {'G': k[0], 'X': k[4]}",
                f"warnings.simplefilter({error_filter})",
                "KPath(kpoints=k, segment_names=n, special_kpoint_map=m)",
            ]
        )
    )
    path = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])

    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": path},
        check=False,
    )

    assert run.returncode == returncode
    assert expected_line.format(script=script, message=MESSAGE) in run.stderr.splitlines()
    assert "DeprecationWarning" not in run.stderr


def test_the_default_filter_shows_a_warning_once_per_line():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(3):
            KPath(kpoints=KPOINTS, segment_names=NAMES, special_kpoint_map=SPECIAL)

    assert [(w.category, w.filename, _source(w)) for w in record] == [(UserWarning, __file__, CALL)]


def test_a_warning_through_a_helper_names_the_helpers_line():
    with user_warning(__file__, match=MESSAGE) as record:
        _labelled_kpath()

    assert [_source(w) for w in record] == [f"return {CALL}"]


def test_a_warning_through_fermi_handler_names_the_callers_line(handler):
    with user_warning(__file__, match="Unknown mode: no_such_mode. Using plain mode.") as record:
        line = _next_line()
        handler.plot_fermi_surface(mode="no_such_mode", show=False)

    assert [w.lineno for w in record] == [line]


@pytest.fixture
def library(tmp_path, monkeypatch):
    package = tmp_path / "warn_user_library"
    package.mkdir()
    (package / "points.py").write_text(
        "\n".join(
            [
                "import dataclasses",
                "from pyprocar.utils.log_utils import warn_user",
                "@dataclasses.dataclass",
                "class Point:",
                "    x: int",
                "    def __post_init__(self):",
                "        warn_user('a warning from __post_init__')",
            ]
        )
    )
    (package / "on_import.py").write_text(
        "from pyprocar.utils.log_utils import warn_user\nwarn_user('a warning on import')\n"
    )
    monkeypatch.setattr(log_utils, "_PACKAGE_DIR", str(package) + os.sep)
    monkeypatch.syspath_prepend(str(tmp_path))
    return package.name


def test_a_warning_from_generated_code_names_the_callers_line(library):
    point = importlib.import_module(f"{library}.points").Point

    with user_warning(__file__, match="a warning from __post_init__") as record:
        line = _next_line()
        point(1)

    assert [w.lineno for w in record] == [line]


def test_a_warning_while_a_module_imports_names_the_import_line(library):
    with user_warning(__file__, match="a warning on import") as record:
        line = _next_line()
        importlib.import_module(f"{library}.on_import")

    assert [w.lineno for w in record] == [line]


def test_a_warning_in_an_executor_thread_names_the_line_that_ran_the_call():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(1) as pool:
            pool.submit(ONE_NAME_FOR_TWO_SEGMENTS).result()

    assert [(w.category, Path(w.filename).parts[-3:]) for w in record] == [
        (UserWarning, ("concurrent", "futures", "thread.py"))
    ]


def test_a_warning_in_a_thread_names_the_line_that_ran_its_target():
    thread = threading.Thread(target=ONE_NAME_FOR_TWO_SEGMENTS)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        thread.start()
        thread.join()

    assert [(w.category, Path(w.filename).name, _source(w)) for w in record] == [
        (UserWarning, "threading.py", "self._target(*self._args, **self._kwargs)")
    ]
