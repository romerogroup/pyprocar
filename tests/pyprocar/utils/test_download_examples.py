import fnmatch
import shutil
import zipfile

import pytest

import pyprocar.utils.download_examples as download_examples

DATASET = {"data/codes.zip": "codes", "data/codes-extra.zip": "extra"}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    downloaded: list[str] = []

    def snapshot_download(repo_id, repo_type, allow_patterns):
        snapshot = tmp_path / "hub" / "datasets--fake" / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        for name, body in DATASET.items():
            if any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                downloaded.append(name)
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(snapshot / name, "w") as z:
                    z.writestr("qe/scf.out", body)
        return str(snapshot)

    monkeypatch.setattr(download_examples, "snapshot_download", snapshot_download)
    out = tmp_path / "out"
    (out / ".cache").mkdir(parents=True)
    (out / ".cache" / "keep").write_text("mine")
    yield out, downloaded
    shutil.rmtree(tmp_path / "hub", ignore_errors=True)


def test_download_from_hf_fetches_only_the_archive_named_exactly(hub):
    out, downloaded = hub

    download_examples.download_from_hf("data/codes", output_path=out)

    assert downloaded == ["data/codes.zip"]
    assert (out / "data/codes/qe/scf.out").read_text() == "codes"
    assert sorted(p.name for p in (out / "data").iterdir()) == ["codes"]
    assert (out / ".cache" / "keep").read_text() == "mine"


def test_download_from_hf_refuses_a_prefix_without_downloading(hub):
    out, downloaded = hub

    with pytest.raises(FileNotFoundError, match=r"data/c\.zip is not in the"):
        download_examples.download_from_hf("data/c", output_path=out)

    assert downloaded == []
    assert not (out / "data").exists()
