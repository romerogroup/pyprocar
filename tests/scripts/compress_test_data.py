from pyprocar.utils.download_examples import compress_dirpath
from tests.scripts.utils import DATA_DIR

if __name__ == "__main__":
    compare_bands_dir = DATA_DIR / "examples" / "00-band_structure" / "compare_bands"
    compress_dirpath(compare_bands_dir)
