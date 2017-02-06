from os import environ, makedirs
from os.path import (
    abspath,
    expanduser,
    exists,
    isfile,
    join,
)
import requests


def get_data_home(data_home=None):
    """Return the path of the cbar data dir.

    The data dir is set to a folder named 'cbar_data' in the user's home
    folder, by default.

    Alternatively, it can be set by the 'CBAR_DATA' environment
    variable or programmatically by providing an explicit folder path. The
    '~' symbol is expanded to the user's home folder.

    The folder is automatically created if it does not exist already .
    """
    if data_home is None:
        data_home = environ.get('CBAR_DATA',
                                join('~', 'cbar_data'))
    data_home = expanduser(data_home)
    create_dir(data_home)
    return data_home


def get_data_dir(data_home, dest_subdir):
    if data_home is None:
        data_dir = join(get_data_home(), dest_subdir)
    else:
        data_dir = join(abspath(data_home), dest_subdir)
    create_dir(data_dir)
    return data_dir


def get_data(url, data_dir, dest_filename, download_if_missing):

    dest_path = join(data_dir, dest_filename)

    if not isfile(dest_path):
        if download_if_missing:
            download(url, dest_path)
        else:
            raise IOError('Dataset missing.')
    return dest_path


def get_quantized_data_path(data_dir, codebook_size):
    return join(data_dir, 'cb_{}_sparse.pkl'.format(codebook_size))


def create_dir(path):
    if not exists(path):
        makedirs(path)


def download(url, dest_path):
    req = requests.get(url, stream=True)
    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)
