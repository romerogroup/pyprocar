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


def compress_test_data(
    test_data_path: Union[str, Path], output_path: Union[str, Path] = None
):
    """Compress test data directory into a zip archive.

    Args:
        test_data_path (str): Path to the test data directory to compress
    """
    import zipfile
    from pathlib import Path

    data_path = Path(test_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_path}")

    archive_path = output_path or data_path.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in data_path.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(data_path.parent))


def upload_test_data_to_hf():

    compress_test_data(DATA_DIR)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATA_DIR.with_suffix(".zip"),
        path_in_repo="data.zip",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    # api.upload_folder(
    #     folder_path=str(TEST_DIR),
    #     path_in_repo="data.zip",
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     ignore_patterns=["**/logs/*.txt", ".cache/*"],
    # )


if __name__ == "__main__":

    upload_test_data_to_hf()
