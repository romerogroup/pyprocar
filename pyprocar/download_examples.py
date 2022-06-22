import gdown


def download_examples():
    gdown.download(id="1AAcJ17ghTVcw_nRICX5IAwhqmtaRP6fd", output='examples.zip')
    gdown.extractall('examples.zip', to = 'examples/')

