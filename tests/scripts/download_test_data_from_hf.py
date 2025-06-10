import os
import shutil
import sys
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR = CURRENT_DIR.parent.parent
TEST_DIR = ROOT_DIR / "tests"
DATA_DIR = TEST_DIR / "data"

sys.path.append(str(ROOT_DIR))


REPO_ID = "lllangWV/pyprocar_test_data"
REPO_TYPE = "dataset"


def download_test_data():
    # snapshot_download(
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     local_dir=DATA_DIR,
    # )

    hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="data.zip",
        local_dir=TEST_DIR,
    )

    uncompress_test_data(TEST_DIR / "data.zip")
    os.remove(TEST_DIR / "data.zip")
    shutil.rmtree(TEST_DIR / ".cache")
    shutil.rmtree(DATA_DIR / ".cache")
    os.remove(DATA_DIR / ".gitattributes")
    os.remove(DATA_DIR / "README.md")


def uncompress_test_data(test_data_path: Union[str, Path]):
    """Uncompress test data from a zip archive.

    Args:
        test_data_path (str): Path to the compressed test data archive
    """
    import zipfile
    from pathlib import Path

    archive_path = Path(test_data_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Test data archive not found: {test_data_path}")

    with zipfile.ZipFile(archive_path, "r") as zipf:
        zipf.extractall(path=archive_path.parent)


if __name__ == "__main__":
    # compress_test_data(DATA_DIR)
    download_test_data()
    # upload_test_data_to_hf()
