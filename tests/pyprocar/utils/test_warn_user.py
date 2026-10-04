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
from tests.pyprocar.scripts.test_fermi_handler_save import handler  # noqa: F401
from tests.utils.user_warning import user_warning

ROOT = Path(__file__).resolve().parents[3]
KPOINTS = np.vstack(
    [np.linspace([0, 0, 0], [0.5, 0, 0], 5), np.linspace([0.5, 0.5, 0.5], [0.5, 0.5, 0], 5)]
)
ONE_NAME_FOR_TWO_SEGMENTS = {
    "kpoints": KPOINTS,
    "segment_names": [("G", "X")],
    "special_kpoint_map": {"G": KPOINTS[0], "X": KPOINTS[4]},
}
MESSAGE = "KPath got 1 segment names for 2 segments in the k-points; ticks use 1"
USER_CODE = "x = 1\nKPath(**ONE_NAME_FOR_TWO_SEGMENTS)\n"


def _next_line() -> int:
    return sys._getframe(1).f_lineno + 1


def _source(warning: warnings.WarningMessage) -> str:
    return linecache.getline(warning.filename, warning.lineno).strip()


def _record(action: str, call) -> list[tuple[str, str, int]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter(action)
        call()
    return [(w.category.__name__, w.filename, w.lineno) for w in record]


def _namespace(**names: object) -> dict[str, object]:
    return {"KPath": KPath, "ONE_NAME_FOR_TWO_SEGMENTS": ONE_NAME_FOR_TWO_SEGMENTS, **names}


def _script_namespace(script: Path) -> dict[str, object]:
    return _namespace(
        __name__="__main__",
        __spec__=None,
        __file__=str(script),
        __loader__=SourceFileLoader("__main__", str(script)),
        __builtins__=__builtins__,
    )


def _labelled_kpath() -> KPath:
    return KPath(**ONE_NAME_FOR_TWO_SEGMENTS)


def test_a_notebook_cell_gets_the_warning_at_its_own_line():
    notebook = vars(types.ModuleType("__main__"))
    notebook.update(_namespace())
    cell = compile(USER_CODE, "<ipython-input-1-abc>", "exec")

    recorded = _record("always", lambda: exec(cell, notebook))

    assert recorded == [("UserWarning", "<ipython-input-1-abc>", 2)]


def test_a_script_gets_the_warning_at_its_own_line_and_no_other_warning(tmp_path):
    script = tmp_path / "user.py"
    script.write_text(USER_CODE)
    code = compile(USER_CODE, str(script), "exec")

    recorded = _record("always", lambda: exec(code, _script_namespace(script)))

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
        ('"error", DeprecationWarning', 0, "{script}:7: UserWarning: {message}"),
    ],
)
def test_a_python_script_sees_the_user_warning_under_an_error_filter(
    tmp_path, error_filter, returncode, expected_line
):
    script = tmp_path / "user.py"
    script.write_text(
        "import warnings\n"
        "import numpy as np\n"
        "from pyprocar.core.kpoints import KPath\n"
        "k = np.vstack([np.linspace([0, 0, 0], [.5, 0, 0], 5), np.linspace([.5, .5, .5], [.5, .5, 0], 5)])\n"
        "n, m = [('G', 'X')], {'G': k[0], 'X': k[4]}\n"
        f"warnings.simplefilter({error_filter})\n"
        "KPath(kpoints=k, segment_names=n, special_kpoint_map=m)\n"
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
            line = _next_line()
            KPath(**ONE_NAME_FOR_TWO_SEGMENTS)

    assert [(w.category, w.filename, w.lineno) for w in record] == [(UserWarning, __file__, line)]


def test_a_warning_through_a_helper_names_the_helpers_line():
    with user_warning(__file__, match=MESSAGE) as record:
        _labelled_kpath()

    assert [_source(w) for w in record] == ["return KPath(**ONE_NAME_FOR_TWO_SEGMENTS)"]


def test_a_warning_through_fermi_handler_names_the_callers_line(handler):  # noqa: F811
    with user_warning(__file__, match="Unknown mode: no_such_mode. Using plain mode.") as record:
        line = _next_line()
        handler.plot_fermi_surface(mode="no_such_mode", show=False)

    assert [w.lineno for w in record] == [line]


@pytest.fixture
def library(tmp_path, monkeypatch):
    package = tmp_path / "warn_user_library"
    package.mkdir()
    (package / "points.py").write_text(
        "import dataclasses\n"
        "from pyprocar.utils.log_utils import warn_user\n"
        "@dataclasses.dataclass\n"
        "class Point:\n"
        "    x: int\n"
        "    def __post_init__(self):\n"
        "        warn_user('a warning from __post_init__')\n"
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
            pool.submit(functools.partial(KPath, **ONE_NAME_FOR_TWO_SEGMENTS)).result()

    assert [(w.category, Path(w.filename).parts[-3:]) for w in record] == [
        (UserWarning, ("concurrent", "futures", "thread.py"))
    ]


def test_a_warning_in_a_thread_names_the_line_that_ran_its_target():
    thread = threading.Thread(target=functools.partial(KPath, **ONE_NAME_FOR_TWO_SEGMENTS))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        thread.start()
        thread.join()

    assert [(w.category, Path(w.filename).name, _source(w)) for w in record] == [
        (UserWarning, "threading.py", "self._target(*self._args, **self._kwargs)")
    ]
