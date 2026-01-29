from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "lllangWV/pyprocar_test_data"
REPO_TYPE = "dataset"


def compress_dirpath(dirpath: str | Path, output_path: str | Path | None = None) -> None:
    """Compress test data directory into a zip archive.

    Parameters
    ----------
        dirpath (Union[str, Path]): Path to the test data directory to compress
        output_path (Union[str, Path]): Path to the compressed test data archive
    """
    dirpath = Path(dirpath)
    if not dirpath.exists():
        raise FileNotFoundError(f"Test data directory not found: {dirpath}")

    archive_path = output_path or dirpath.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in dirpath.rglob("*"):
            zipf.write(file, file.relative_to(dirpath.parent))


def uncompress_dirpath(dirpath: str | Path) -> None:
    """Uncompress test data from a zip archive.

    Parameters
    ----------
        dirpath (Union[str, Path]): Path to the compressed test data archive.
    """
    archive_path = Path(dirpath)
    if archive_path.suffix == ".zip":
        outpath = archive_path.with_suffix("")
    else:
        outpath = archive_path

    if not archive_path.exists():
        raise FileNotFoundError(f"Test data archive not found: {dirpath}")

    with zipfile.ZipFile(archive_path, "r") as zipf:
        zipf.extractall(path=outpath)


def compress_test_data(data_dirpath: str | Path) -> None:
    """Compress test data with custom logic for different directories.

    - codes, io, issues directories are compressed as whole directories
    - examples directory: compress directories that are 3 levels deep
    """
    data_dirpath = Path(data_dirpath)
    codes_dirpath = data_dirpath / "codes"
    examples_dirpath = data_dirpath / "examples"
    io_dirpath = data_dirpath / "io"
    issues_dirpath = data_dirpath / "issues"

    # Compress codes, io, and issues directories as whole directories
    for dp in [codes_dirpath, io_dirpath, issues_dirpath]:
        if dp.exists() and dp.is_dir():
            print(f"Compressing {dp.name} directory...")
            compress_dirpath(dp)

    # For examples directory, compress directories that are 3 levels deep
    if examples_dirpath.exists() and examples_dirpath.is_dir():
        print("Compressing examples subdirectories...")
        for level1_dir in examples_dirpath.iterdir():
            if level1_dir.is_dir():
                for level2_dir in level1_dir.iterdir():
                    print(f"Compressing {level2_dir.relative_to(data_dirpath)}...")
                    if "zip" in level2_dir.name:
                        continue
                    if level2_dir.is_dir():
                        shutil.make_archive(str(level2_dir), "zip", str(level2_dir))
                    else:
                        compress_dirpath(level2_dir)


def uncompress_test_data(data_dirpath: str | Path) -> None:
    """Uncompress test data with custom logic for different directories.

    - codes, io, issues directories are uncompressed from whole directory zip files
    - examples directory: uncompress directories that were compressed at level 2

    Parameters
    ----------
        data_dirpath (Union[str, Path]): Path to the directory containing compressed test data
    """
    data_dirpath = Path(data_dirpath)

    # Uncompress codes, io, and issues directories from whole directory zip files
    for dir_name in ["codes", "io", "issues"]:
        zip_path = data_dirpath / f"{dir_name}.zip"
        if zip_path.exists():
            print(f"Uncompressing {dir_name} directory...")
            uncompress_dirpath(zip_path)
            # Remove the zip file after extraction
            zip_path.unlink()

    # For examples directory, uncompress directories that were compressed at level 2
    examples_dirpath = data_dirpath / "examples"
    if examples_dirpath.exists() and examples_dirpath.is_dir():
        print("Uncompressing examples subdirectories...")
        for level1_dir in examples_dirpath.iterdir():
            if level1_dir.is_dir():
                for level2_dir in level1_dir.iterdir():
                    if level2_dir.is_file() and level2_dir.suffix == ".zip":
                        print(f"Uncompressing {level2_dir.relative_to(data_dirpath)}...")
                        # Create directory with same name as zip (without .zip extension)
                        extract_dir = level2_dir.with_suffix("")
                        extract_dir.mkdir(exist_ok=True)

                        # Extract the zip file to the new directory
                        with zipfile.ZipFile(level2_dir, "r") as zip_ref:
                            zip_ref.extractall(extract_dir)

                        # Remove the zip file after extraction
                        level2_dir.unlink()


def download_test_data(relpath: str, output_path: str | Path = ".", force: bool = False) -> Path:
    """Download test data from HuggingFace Hub.

    Parameters
    ----------
        relpath (str): Path to the file to download from the Hugging Face Hub.
        This should be relative to the root `data/issues` directory.

        output_path (str): Path to the directory to download the examples to.
    """
    output_path = Path(output_path)
    full_data_path = output_path / relpath
    if full_data_path.exists() and not force:
        print(f"Data already exists at {full_data_path}")
        return full_data_path

    # Ensure the output directory exists - this is critical for Jupyter notebooks
    output_path.mkdir(parents=True, exist_ok=True)

    pattern = relpath + "*"

    download_dirpath = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=[pattern],
    )
    download_path = Path(download_dirpath)
    data_dir = download_path / "data"

    dataset_cache_dir = download_path.parent.parent

    shutil.copytree(data_dir, output_path / "data", dirs_exist_ok=True)
    shutil.rmtree(dataset_cache_dir)

    data_dir = output_path / "data"
    # print(data_dir)
    # uncompress_test_data(data_dir)

    uncompress_dirpath(full_data_path.with_suffix(".zip"))

    os.remove(full_data_path.with_suffix(".zip"))

    # Handle cache directory cleanup more safely
    cache_dir = output_path / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    return full_data_path


def download_from_hf(relpath: str, output_path: str | Path = ".", force: bool = False) -> Path:
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(download_test_data, relpath, output_path, force)
        return future.result()


def remove_zip_files(dirpath: str | Path) -> None:
    """Remove all zip files from a directory and its subdirectories.

    Parameters
    ----------
    dirpath : Union[str, Path]
        Path to the directory from which to remove all zip files recursively.

    Examples
    --------
    >>> remove_zip_files("./my_directory")
    >>> remove_zip_files(Path("./data"))
    """
    dirpath = Path(dirpath)

    if not dirpath.exists():
        raise FileNotFoundError(f"Directory not found: {dirpath}")

    if not dirpath.is_dir():
        raise ValueError(f"Path is not a directory: {dirpath}")

    # Find all zip files recursively
    zip_filepaths = list(dirpath.rglob("*.zip"))

    if not zip_filepaths:
        print(f"No zip files found in {dirpath}")
        return

    print(f"Found {len(zip_filepaths)} zip files to remove...")

    # Remove each zip file
    for filepath in zip_filepaths:
        try:
            filepath.unlink()
            print(f"Removed: {filepath.relative_to(dirpath)}")
        except Exception as e:
            print(f"Failed to remove {filepath.relative_to(dirpath)}: {e}")

    print(f"Finished removing zip files from {dirpath}")


def upload_test_data_to_hf(data_dirpath: str | Path) -> None:
    """Upload test data to Hugging Face Hub.

    Compresses the test data directory and uploads it to the Hugging Face Hub.

    Parameters
    ----------
        data_dirpath (Union[str, Path]): Path to the directory containing test data to upload
    """
    compress_test_data(data_dirpath)
    api = HfApi()
    api.upload_folder(
        folder_path=str(data_dirpath),
        path_in_repo="data",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=["*.zip"],
    )

    remove_zip_files(data_dirpath)
