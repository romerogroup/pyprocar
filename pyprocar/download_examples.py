import os
import gdown

# TODO Zip file in google drive then download
# TODO Save dir path to temp
def download_examples(save_dir=''):
    if save_dir != '':
        output = f"{save_dir}{os.sep}examples.zip"
        to = f"{save_dir}{os.sep}examples{os.sep}"
    else:
        output='examples.zip'
        to = f"examples{os.sep}"
        
    gdown.download(id="1AAcJ17ghTVcw_nRICX5IAwhqmtaRP6fd", output=output)
    gdown.extractall(output, to = to)

