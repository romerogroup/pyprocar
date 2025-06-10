from utils import DATA_DIR, TEST_DIR

from pyprocar.utils.download_examples import download_test_data

if __name__ == "__main__":
    download_test_data("data", output_path=TEST_DIR)
