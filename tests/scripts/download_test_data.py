from pyprocar.utils.download_examples import download_test_data
from tests.scripts.utils import TEST_DIR

if __name__ == "__main__":
    download_test_data("data", output_path=TEST_DIR)
