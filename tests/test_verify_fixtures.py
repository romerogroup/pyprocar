import hashlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(".claude/skills/verify-pyprocar/scripts/verify.sh")
# VERIFY_SH_UNDER_TEST runs the suite against another copy of verify.sh, such as a mutant.
VERIFY_SH = Path(
    os.environ.get("VERIFY_SH_UNDER_TEST") or Path(__file__).resolve().parents[1] / SCRIPT
)
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
FAKE_ENV = """#!/bin/sh
echo "cleanup=${{HF_XET_LOG_DIR_DISABLE_CLEANUP-unset}} $@" >> "{log}"
for last; do :; done
case "$last" in data/*) mkdir -p "$last" && echo x > "$last/PROCAR" ;; esac
"""


@dataclass(frozen=True, slots=True)
class Harness:
    root: Path
    repo: Path
    data_rel: str
    shared_env: Path | None

    @property
    def env_log(self) -> Path:
        return self.root / "env.log"

    @property
    def data_dir(self) -> Path:
        return self.root / self.data_rel

    def verify(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}"}
        return subprocess.run(
            ["bash", str(self.repo / SCRIPT), *args],
            cwd=self.repo,
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

    def install_shared_env(self) -> None:
        assert self.shared_env is not None
        _stub(self.shared_env / "python", self.env_log)

    def locked(self) -> set[str]:
        return {
            p.relative_to(self.root).as_posix()
            for p in _entries(self.root)
            if not p.lstat().st_mode & WRITE_BITS
        }

    def fixture_lock(self, rel: str) -> set[str]:
        return {p.relative_to(self.root).as_posix() for p in _entries(self.data_dir / rel)}

    def lock(self, rel: str) -> None:
        for p in _entries(self.data_dir / rel):
            p.chmod(p.lstat().st_mode & ~WRITE_BITS)


def _entries(root: Path) -> list[Path]:
    return [p for p in [root, *root.rglob("*")] if not p.is_symlink()]


def _stub(path: Path, log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_ENV.format(log=log))
    path.chmod(0o755)


def _git(*args: str | Path) -> None:
    subprocess.run(["git", *map(str, args)], check=True, capture_output=True)


@pytest.fixture(params=["data_in_repo", "worktree"])
def harness(tmp_path, request):
    main = tmp_path / "repo"
    (main / SCRIPT).parent.mkdir(parents=True)
    shutil.copy(VERIFY_SH, main / SCRIPT)
    _git("init", "-q", main)
    data = main / "data"
    repo, shared_env = main, None
    if request.param == "worktree":
        _git("-C", main, "add", SCRIPT)
        _git("-C", main, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
        repo = tmp_path / "wt"
        _git("-C", main, "worktree", "add", "-q", repo)
        (repo / "data").symlink_to(data)
        shared_env = main / ".pixi/envs/dev/bin"
    for rel in ("examples/bands/x/PROCAR", "codes/qe/scf.out", "verify-runs/run1/evidence/log"):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text("x")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "f").write_text("x")
    for rel in OUTSIDE:
        (tmp_path / "outside" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "outside" / rel).write_text("x")
    (data / "escape").symlink_to(os.path.relpath(victim, data))
    (data / "runs").symlink_to("verify-runs")
    (data / "self").symlink_to(".")
    (data / "examples/bands/x/outside").symlink_to(victim)
    (data / "examples/bands/dangling").symlink_to(victim / "missing")
    (data / "examples/bands/loop").symlink_to("loop")
    harness = Harness(tmp_path, repo, data.relative_to(tmp_path).as_posix(), shared_env)
    _stub(tmp_path / "bin" / "pixi", harness.env_log)
    for p in _entries(victim):
        p.chmod(p.stat().st_mode & ~WRITE_BITS)
    (tmp_path / "outside/unenterable").chmod(0)
    yield harness
    for p in _entries(tmp_path):
        p.chmod(p.lstat().st_mode | stat.S_IRWXU)


OUTSIDE = ("home/f", "tmp/runs/x/work/f", "tmp/runs/x/evidence/log", "unenterable/f")


REFUSED = [
    "data",
    "data/",
    "data/examples",
    "data/examples/bands",
    "data/verify-runs",
    "data/verify-runs/run1",
    "examples/bands/x",
    "./data/codes",
    "{repo}/data/codes",
    "data/..",
    "data/../victim",
    "data/examples/bands/../../../victim",
    "data/codes/qe/..",
    "data/.",
    "data/./verify-runs",
    "data/./codes",
    "data//codes",
    "data/escape",
    "data/runs",
    "data/runs/run1",
    "data/self",
    "data/*",
    "data/examples/bands/*",
    "data/examples/*/x",
    "data/c?des",
    "data/[c]odes",
    "data/examples/bands/x*",
    "data/-R",
    "data/codes/qe",
    "data/examples/bands/x/outside",
    "data/examples/bands/dangling",
    "data/examples/bands/loop",
]


@pytest.mark.parametrize("rel", REFUSED)
def test_fetch_refuses_a_path_that_is_not_a_fixture_under_data(harness, rel):
    before = harness.locked()

    assert harness.verify("fetch", rel.format(repo=harness.repo)).returncode == 2
    assert harness.locked() == before
    assert not harness.env_log.exists()


@pytest.mark.parametrize(
    ("rel", "fixture"),
    [
        ("data/examples/bands/x", ["examples/bands/x", "examples/bands/x/PROCAR"]),
        ("data/examples/bands/x/", ["examples/bands/x", "examples/bands/x/PROCAR"]),
        ("data/codes", ["codes", "codes/qe", "codes/qe/scf.out"]),
    ],
)
def test_fetch_locks_an_existing_fixture_without_the_env(harness, rel, fixture):
    before = harness.locked()

    assert harness.verify("fetch", rel).returncode == 0
    assert harness.locked() - before == {f"{harness.data_rel}/{p}" for p in fixture}
    assert not harness.env_log.exists()


def test_fetch_through_a_symlinked_fixture_locks_its_target_inside_data(harness):
    (harness.data_dir / "examples/bands/alias").symlink_to("../../codes")
    before = harness.locked()

    assert harness.verify("fetch", "data/examples/bands/alias").returncode == 0
    assert harness.locked() - before == {
        f"{harness.data_rel}/{p}" for p in ["codes", "codes/qe", "codes/qe/scf.out"]
    }


def test_fetch_downloads_a_missing_fixture_by_its_literal_name_then_locks_it(harness):
    _prime(harness)
    before = harness.locked()

    assert harness.verify("fetch", "data/examples/dos/new").returncode == 0
    logged = harness.env_log.read_text().split()
    assert logged[0] == "cleanup=1" and logged[-1] == "data/examples/dos/new"
    if harness.shared_env is None:
        assert logged[1:5] == ["run", "-q", "--locked", "-e"]
    assert harness.locked() - before == {
        f"{harness.data_rel}/examples/dos/new",
        f"{harness.data_rel}/examples/dos/new/PROCAR",
    }


def test_fetch_in_a_worktree_without_the_env_refuses_only_a_download(harness):
    if harness.shared_env is None:
        pytest.skip("the env check applies to a linked worktree")
    before = harness.locked()

    assert harness.verify("fetch", "data/examples/dos/new").returncode == 2
    assert harness.locked() == before
    assert harness.verify("fetch", "data/codes").returncode == 0


def _prime(harness) -> None:
    if harness.shared_env is not None:
        harness.install_shared_env()
    (harness.root / "driver.py").write_text("print('probe')\n")


def _run(harness, fixture, name="probe"):
    _prime(harness)
    return harness.verify("run", name, fixture, str(harness.root / "driver.py"))


def _calc(harness) -> Path:
    (run,) = (harness.data_dir / "verify-runs").glob("*-probe")
    return run / "work" / "calc"


@pytest.mark.parametrize("link", ["absolute", "relative"])
def test_run_copies_a_symlinked_fixture_and_leaves_the_fixture_locked(harness, link):
    harness.lock("examples/bands/x")
    target = harness.data_dir / "examples/bands/x"
    (harness.data_dir / "scratchlinks").mkdir()
    (harness.data_dir / "scratchlinks/x").symlink_to(
        target if link == "absolute" else Path("../examples/bands/x")
    )
    before = harness.locked()

    assert _run(harness, "data/scratchlinks/x").returncode == 0
    assert harness.locked() == before
    calc = _calc(harness)
    assert not calc.is_symlink() and not (calc / "outside").is_symlink()
    (calc / "PROCAR").write_text("rewritten")
    (calc / "outside" / "f").write_text("rewritten")
    assert (target / "PROCAR").read_text() == "x"
    assert (harness.root / "victim" / "f").read_text() == "x"


@pytest.mark.parametrize(
    "fixture",
    ["data/escape", "{root}/victim", "../victim", "data/verify-runs/run1", "data/*", "data/runs"],
)
def test_run_refuses_a_fixture_outside_the_fixtures(harness, fixture):
    before = harness.locked()

    assert _run(harness, fixture.format(root=harness.root)).returncode == 2
    assert harness.locked() == before
    assert sorted(p.name for p in (harness.data_dir / "verify-runs").iterdir()) == ["run1"]


@pytest.mark.parametrize("name", ["a/b", "../../examples/bands/x", "a/../../../x", ""])
def test_run_refuses_a_name_that_is_not_one_plain_component(harness, name):
    assert _run(harness, "data/examples/bands/x", name=name).returncode == 2
    assert sorted(p.name for p in (harness.data_dir / "verify-runs").iterdir()) == ["run1"]


def _plant_the_next_run(harness, how: str) -> None:
    (harness.root / "bin/date").write_text("#!/bin/sh\necho 20260101-000000\n")
    (harness.root / "bin/date").chmod(0o755)
    run = harness.data_dir / "verify-runs/20260101-000000-probe"
    victim = harness.root / "outside/runs/victim"
    (victim / "work/calc/x").mkdir(parents=True)
    (victim / "work/calc/x/PROCAR").write_text("victim")
    (victim / "evidence").mkdir()
    (victim / ".pid").write_text("1")
    if how == "run_dir_link":
        run.symlink_to(victim)
        return
    run.mkdir()
    if how != "run_dir":
        (run / how).symlink_to(victim / how)


@pytest.mark.parametrize("how", ["run_dir_link", "run_dir", "work", ".pid", "evidence"])
def test_run_refuses_a_run_dir_that_already_exists_and_changes_nothing(harness, how):
    _plant_the_next_run(harness, how)
    _prime(harness)
    before = _snapshot(harness)

    out = _run(harness, "data/examples/bands/x")

    assert out.returncode == 2, out.stderr
    assert not harness.env_log.exists()
    assert _snapshot(harness) == before


def _run_dir(harness, name: str) -> Path:
    run = harness.data_dir / "verify-runs" / name
    (run / "evidence").mkdir(parents=True)
    (run / "evidence" / "exit_code").write_text("0")
    return run


def test_clean_removes_a_read_only_work_copy_and_keeps_evidence(harness):
    run = _run_dir(harness, "r")
    (run / "work/calc/sub").mkdir(parents=True)
    (run / "work/calc/sub/PROCAR").write_text("x")
    harness.lock("verify-runs/r/work")

    assert harness.verify("clean", str(run)).returncode == 0
    assert not (run / "work").exists()
    assert (run / "evidence/exit_code").read_text() == "0"


@pytest.mark.parametrize(
    "target",
    ["data/examples/bands/x", "data/verify-runs", "data/verify-runs/r/evidence", "{root}/victim"],
)
def test_clean_refuses_a_path_that_is_not_a_run_directory(harness, target):
    _run_dir(harness, "r")
    (harness.data_dir / "examples/bands/x/work").mkdir()
    before = harness.locked()

    assert harness.verify("clean", target.format(root=harness.root)).returncode == 2
    assert harness.locked() == before
    assert (harness.data_dir / "examples/bands/x/work").is_dir()


def test_clean_removes_read_only_work_and_never_unlocks_a_fixture_it_links_to(harness):
    harness.lock("examples/bands/x")
    linked = _run_dir(harness, "linked")
    (linked / "work").symlink_to(harness.data_dir / "examples/bands/x")
    inner = _run_dir(harness, "inner")
    (inner / "work/tmp").mkdir(parents=True)
    (inner / "work/calc").symlink_to(harness.data_dir / "examples/bands/x")
    harness.lock("verify-runs/inner/work")
    before = harness.locked()

    assert harness.verify("clean", str(linked)).returncode == 0
    assert harness.verify("clean", str(inner)).returncode == 0
    assert harness.locked() >= harness.fixture_lock("examples/bands/x")
    assert harness.locked() - before == set()
    assert not (linked / "work").is_symlink() and not (inner / "work").exists()
    assert (harness.data_dir / "examples/bands/x/PROCAR").read_text() == "x"


def test_gc_removes_read_only_work_and_never_unlocks_a_fixture_it_links_to(harness):
    harness.lock("examples/bands/x")
    run = _run_dir(harness, "old")
    (run / "work/tmp").mkdir(parents=True)
    (run / "work/calc").symlink_to(harness.data_dir / "examples/bands/x")
    hour_ago = (run / "work").stat().st_mtime - 3600
    os.utime(run / "work", (hour_ago, hour_ago))
    harness.lock("verify-runs/old/work")
    before = harness.locked()

    assert harness.verify("gc", "0").returncode == 0
    assert harness.locked() >= harness.fixture_lock("examples/bands/x")
    assert harness.locked() - before == set()
    assert not (run / "work").exists()
    assert (harness.data_dir / "examples/bands/x/PROCAR").read_text() == "x"


@pytest.mark.guards_existing_behaviour(reason="dev's gc already keeps a run whose pid is alive")
def test_gc_keeps_the_work_of_a_run_in_progress_and_removes_a_finished_one(harness):
    finished = subprocess.Popen(["true"])
    finished.wait()
    runs = {}
    for name, pid in (("live", os.getpid()), ("done", finished.pid)):
        runs[name] = _run_dir(harness, name)
        (runs[name] / "work/calc").mkdir(parents=True)
        (runs[name] / ".pid").write_text(str(pid))
        hour_ago = (runs[name] / "work").stat().st_mtime - 3600
        os.utime(runs[name] / "work", (hour_ago, hour_ago))

    out = harness.verify("gc", "0")

    assert out.returncode == 0, out.stderr
    assert (runs["live"] / "work/calc").is_dir()
    assert not (runs["done"] / "work").exists()
    live_work = runs["live"].resolve() / "work"
    assert f"keeping {live_work} (run in progress, pid {os.getpid()})" in out.stdout.splitlines()


def _doctor_lines(harness) -> list[str]:
    _prime(harness)
    out = harness.verify("doctor")
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def _doctor_count(harness) -> int:
    (line,) = [x for x in _doctor_lines(harness) if x.startswith("writable fixture paths:")]
    return int(line.split(":")[1])


def _unlockable(harness) -> list[str]:
    lines = _doctor_lines(harness)
    (i,) = [n for n, x in enumerate(lines) if x.startswith("unlockable fixture roots:")]
    return [x.strip() for x in lines[i + 1 : i + 1 + int(lines[i].split(":")[1])]]


def test_doctor_counts_each_writable_fixture_path_once(harness):
    (harness.data_dir / "examples/bands/alias").symlink_to("../../codes")

    assert _doctor_count(harness) == 5
    assert harness.verify("fetch", "data/codes").returncode == 0
    assert _doctor_count(harness) == 2
    assert harness.verify("fetch", "data/examples/bands/x").returncode == 0
    assert _doctor_count(harness) == 0


def test_doctor_counts_a_symlinked_fixture_root_by_its_target(harness):
    hidden = harness.data_dir / ".stash/fix"
    hidden.mkdir(parents=True)
    (hidden / "OUTCAR").write_text("x")
    (harness.data_dir / "examples/bands/alias").symlink_to("../../.stash/fix")
    harness.lock("examples/bands/x")
    harness.lock("codes")

    assert _doctor_count(harness) == 2


def test_doctor_lists_the_fixture_roots_that_fetch_refuses_to_lock(harness):
    for name in (".hidden", "two words"):
        (harness.data_dir / name).mkdir()
        (harness.data_dir / name / "f").write_text("x")

    assert _unlockable(harness) == [
        "data/.hidden",
        "data/escape",
        "data/examples/bands/dangling",
        "data/examples/bands/loop",
        "data/runs",
        "data/self",
        "data/two words",
    ]
    assert harness.verify("fetch", "data/.hidden").returncode == 2
    assert harness.verify("fetch", "data/two words").returncode == 2


def _snapshot(harness, *allowed: Path) -> dict[str, tuple[int, str]]:
    skip = {harness.env_log, *allowed}
    snap: dict[str, tuple[int, str]] = {}
    for top, dirs, files in os.walk(harness.root):
        dirs[:] = [d for d in dirs if Path(top, d) not in skip and d != ".git"]
        for path in (Path(top, name) for name in [*dirs, *files]):
            if path in skip or path.name == ".git":
                continue
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                body = os.readlink(path)
            elif stat.S_ISREG(mode) and os.access(path, os.R_OK):
                body = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                body = ""
            snap[path.relative_to(harness.root).as_posix()] = (stat.S_IMODE(mode), body)
    return snap


def _outside(harness, *allowed: str) -> dict[str, tuple[int, str]]:
    return _snapshot(harness, harness.data_dir, *(harness.repo / a for a in allowed))


def _relink_data(harness, target: Path | str) -> None:
    data = harness.repo / "data"
    if data.is_symlink():
        data.unlink()
    else:
        data.rename(harness.root / "data-moved")
    data.symlink_to(target)


REFUSING_EVERY_COMMAND = [
    ("fetch", "data/pyprocar-288-absent"),
    ("fetch", "data/codes"),
    ("fetch", "data/examples/bands/x"),
    ("run", "probe", "data/examples/bands/x", "{root}/driver.py"),
    ("clean", "data/verify-runs/run1"),
    ("gc", "0"),
]


def _refuses_everything(harness, *extra: tuple[str, ...]) -> None:
    _prime(harness)
    before = _snapshot(harness)

    for args in [*REFUSING_EVERY_COMMAND, *extra]:
        out = harness.verify(*(a.format(root=harness.root) for a in args))
        assert out.returncode == 2, (args, out.stderr)
    assert not harness.env_log.exists()
    assert harness.verify("doctor").returncode == 2
    assert _snapshot(harness) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters any directory")
@pytest.mark.parametrize("how", ["data_dir_unenterable", "data_links_to_an_unenterable_dir"])
def test_an_unenterable_data_root_refuses_every_command(harness, how):
    assert not any(
        Path("/", n).exists() for n in ("pyprocar-288-absent", "codes", "examples", "verify-runs")
    )
    if how == "data_dir_unenterable":
        harness.data_dir.chmod(0)
    else:
        _relink_data(harness, harness.root / "outside/unenterable")

    _refuses_everything(harness)


def test_a_data_root_at_the_top_of_the_tree_refuses_every_command(harness):
    _relink_data(harness, harness.root)

    _refuses_everything(harness, ("fetch", "data/outside"), ("fetch", "data/repo"))


def test_a_data_link_to_a_copy_outside_the_main_checkout_is_refused(harness):
    shutil.copytree(harness.data_dir, harness.root / "outside/data", symlinks=True)
    _relink_data(harness, harness.root / "outside/data")

    _refuses_everything(harness)


def test_a_worktree_data_link_spelled_with_a_leading_double_slash_still_locks(harness):
    if harness.shared_env is None:
        pytest.skip(
            "a main checkout's data/ is a directory, so only a worktree link has a spelling"
        )
    _relink_data(harness, f"/{harness.data_dir}")
    before = harness.locked()

    assert harness.verify("fetch", "data/codes").returncode == 0
    assert harness.locked() - before == {
        f"{harness.data_rel}/{p}" for p in ["codes", "codes/qe", "codes/qe/scf.out"]
    }


def test_a_verify_runs_link_out_of_data_is_never_followed(harness):
    shutil.rmtree(harness.data_dir / "verify-runs")
    (harness.data_dir / "verify-runs").symlink_to(harness.root / "outside/tmp/runs")
    (harness.root / "outside/tmp/runs/x/work").chmod(0o555)
    _prime(harness)
    before = _snapshot(harness)

    assert harness.verify("clean", "data/verify-runs/x").returncode == 2
    assert _run(harness, "data/examples/bands/x").returncode == 2
    assert harness.verify("gc", "0").returncode == 2
    assert _snapshot(harness) == before


def test_fetch_leaves_a_hard_link_to_a_file_outside_data_writable(harness):
    os.link(harness.root / "outside/home/f", harness.data_dir / "codes/qe/hard")
    harness.lock("examples/bands/x")
    before = _outside(harness)

    out = harness.verify("fetch", "data/codes")

    assert out.returncode == 0
    assert _outside(harness) == before
    assert out.stderr.splitlines()[-1] == f"{harness.root}/{harness.data_rel}/codes/qe/hard"
    assert not (harness.data_dir / "codes/qe/scf.out").stat().st_mode & WRITE_BITS
    assert _doctor_count(harness) == 1


def test_no_command_changes_anything_outside_data(harness):
    os.link(harness.root / "outside/home/f", harness.data_dir / "codes/qe/hard")
    (harness.data_dir / "verify-runs/run1/work").symlink_to(harness.root / "outside/tmp")
    (harness.data_dir / "examples/bands/alias").symlink_to("../../codes")
    (harness.data_dir / ".hidden").mkdir()
    allowed: tuple[str, ...] = ()
    if harness.shared_env is not None:
        (harness.root / "repo/pyprocar").mkdir()
        (harness.root / "repo/pyprocar/_version.py").write_text("version")
        (harness.repo / "pyprocar").mkdir()
        allowed = ("pyprocar", ".tmp", "data")
    _prime(harness)
    before = _outside(harness, *allowed)

    fetched = ["data/codes", "data/examples/bands/alias", "data/examples/dos/new", "data/.hidden"]
    for rel in [*REFUSED, *fetched]:
        harness.verify("fetch", rel.format(repo=harness.repo))
    for fixture in ["data/examples/bands/x", "data/escape", "{root}/outside/home", "data/runs"]:
        _run(harness, fixture.format(root=harness.root))
    for run in [*(harness.data_dir / "verify-runs").iterdir(), harness.root / "outside/tmp/runs/x"]:
        harness.verify("clean", str(run))
    harness.verify("gc", "0")
    harness.verify("doctor")
    setup = harness.verify("worktree-setup")

    assert _outside(harness, *allowed) == before
    if harness.shared_env is not None:
        assert setup.returncode == 0, setup.stderr
        assert (harness.repo / "pyprocar/_version.py").read_text() == "version"
        assert os.readlink(harness.repo / "data") == str(harness.data_dir)
        assert (harness.repo / ".tmp").is_dir()


WRITING_COMMAND = (
    r"\b(chmod|chown|chgrp|chattr|setfacl|rm|rmdir|unlink|mv|shred|cp|ln|mkdir|touch|tee|truncate"
    r"|dd|rsync|mktemp|mkfifo|mknod|install|tar|unzip|download_from_hf)\b|-delete\b"
    r"|\b(sed|perl)\b[^|;]*\s(-[A-Za-z]*i|--in-place)"
)
FILE_REDIRECT = r"(&|\d+|\{\w+\})?(>>|>\||<>|>)(?![>|])(?!\s*/dev/null\b)(?!&(\d+|-))"
QUOTED = r""""(?:[^"\\]|\\.)*"|'[^']*'"""
LONE_EXPANSION = r'"\$(\w+|\{\w+\}|[@*])"'
VARIABLE_COMMAND = (
    r"(^|[;&|({!`]|\$\(|\b(then|do|else|exec|eval|xargs|command|env|nohup|time|sudo)\s)"
    r"\s*(\w+=\S*\s+)*\"?\$(\{|\w|[@*])"
)


def _command_words(line: str) -> str:
    blanked = re.sub(QUOTED, lambda m: m[0] if re.fullmatch(LONE_EXPANSION, m[0]) else "''", line)
    return re.sub(r"\[\[.*?\]\]", "", blanked)


def _census(script: str) -> list[str]:
    return [
        x
        for x in (line.strip() for line in script.splitlines())
        if not x.startswith("#")
        and (
            re.search(WRITING_COMMAND, x)
            or re.search(VARIABLE_COMMAND, _command_words(x))
            or re.search(FILE_REDIRECT, re.sub(QUOTED, "", x))
        )
    ]


@pytest.mark.parametrize(
    "line",
    [
        'exec 3>"$f"',
        '2>"$f" true',
        'echo x 1>"$f"',
        'echo x &>"$f"',
        'echo x >&"$f"',
        'echo x >|"$f"',
        'exec {fd}>>"$f"',
        'sed -i "s/a/b/" "$f"',
        'sed --in-place "s/a/b/" "$f"',
        'perl -pi -e "s/a/b/" "$f"',
        'install "$f" "$g"',
        'tar -xf "$f"',
        'unzip -o "$f"',
        '$w "$f"',
        '"$w" "$f"',
        'X=1 "${w}" "$f"',
        'true && "$@"',
    ],
)
def test_the_census_flags_each_way_a_line_can_write(line):
    script = VERIFY_SH.read_text()
    harmless = 'echo "a > b" >&2 2>/dev/null; exec 3>&-; cmd 2>&1 >>/dev/null; x="$y"'

    assert _census(f"{script}\n{harmless}") == _census(script)
    assert _census(f"{script}\n{line}") == [*_census(script), line]


def test_verify_sh_writes_only_on_the_lines_its_census_reviewed():
    writes = _census(VERIFY_SH.read_text())

    assert writes == [
        'PATH="$SHARED_ENV:$PATH" PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" '
        + 'PYTHONDONTWRITEBYTECODE=1 "$@"',
        '[ -x "$SHARED_ENV/python" ] || { echo "missing $SHARED_ENV/python; '
        + "run 'pixi install -e dev' in $MAIN\" >&2; exit 2; }",
        'find -P "$2" ! -type l \\( -type d -o -links 1 \\) -exec chmod "$1" {} +',
        'rm -rf "$1/work" "$1/.start" "$1/.pid"',
        'pyprocar.download_from_hf(relpath=sys.argv[1], output_path=Path(".").resolve())\' "$rel"',
        'mkdir -p "$RUNS"',
        'mkdir "$run" ||',
        'mkdir "$run/evidence" "$run/work" "$run/work/tmp"',
        '(set -C; echo $$ >"$run/.pid")',
        "trap 'rm -f \"$run/.pid\"' EXIT",
        'cp -RL --reflink=auto "$fixture" "$run/work/calc"',
        'cp "$driver" "$run/evidence/driver.py"',
        'touch "$run/.start"',
        'py "$run/evidence/driver.py" >"$run/evidence/run.log" 2>&1',
        'echo "$code" >"$run/evidence/exit_code"',
        '>"$run/evidence/side_effects.txt"',
        '{ echo "missing $MAIN/pyprocar/_version.py; '
        + "run 'pixi install -e dev' in $MAIN\" >&2; exit 2; }",
        'cp "$MAIN/pyprocar/_version.py" pyprocar/_version.py',
        'mkdir -p "$MAIN/data" .tmp',
        'ln -sfn "$MAIN/data" data',
        "mkdir -p .tmp",
    ]
