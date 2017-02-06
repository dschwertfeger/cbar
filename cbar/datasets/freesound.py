from os import listdir
from os.path import join

import pandas as pd

from .base import get_data_dir
from .base import get_quantized_data_path

from ..preprocess import (
    get_relevant_queries,
    quantize_mfccs,
    preprocess_tags,
    preprocess_queries,
)

FREESOUND_DIR = 'freesound'
SOUNDS = 'sounds.json'
TAGS_FILE = 'preprocessed_tags.csv'

QUERIES = 'queries.csv'
PREPROCESSED_QUERIES = 'preprocessed_queries.csv'

FS_DL_PAGE = ('https://www.kaggle.com/dschwertfeger/freesound/')


def load_freesound(codebook_size, data_home=None, **kwargs):
    r"""Loader for the Freesound dataset [4]_.

    .. warning::

        You need to `download the Freesound dataset from Kaggle
        <https://www.kaggle.com/dschwertfeger/freesound/>`_ and unpack it into
        your home directory or the directory specified as ``data_home`` for
        this loader to work.

    This dataset consists of 227,085 sounds. Each sound is at most 30 seconds
    long and annotated with tags from a tag vocabulary of 3,466 tags. The
    sounds' original tags were provided by the users who uploaded the sounds
    to `Freesound <http://www.freesound.org>`_. The more than 50,000
    original tags were preprocessed to form a tag vocabulary of 3,466 tags.

    Parameters
    ----------
    codebook_size : int, 512, 1024, 2048, 4096
        The codebook size. The dataset is pre-encoded with codebook sizes of
        512, 1024, 2048, and 4096. If you want to experiment with other
        codebook-sizes, you need to download the orginal MFCCs, append the
        first-order and second-order derivatives and quantize the resulting
        frame-vectors specifying the desired ``codebook_size`` using
        :func:`cbar.preprocess.quantize_mfccs`.

    data_home : optional
        Specify a home folder for the Freesound datasets. By default (``None``)
        the files are expected to be in ``~/cbar_data/freesound/``, where
        ``cbar_data`` is the ``data_home`` directory.

    Returns
    -------
    X : pd.DataFrame, shape = [227085, codebook_size]
        Each row corresponds to a preprocessed sound, represented as a sparse
        codebook vector.

    Y : pd.DataFrame, shape = [227085,]
        Tags associated with each sound provided as a list of strings. Use
        :func:`sklearn.preprocessing.MultiLabelBinarizer` to transform tags
        into binary indicator format.

    References
    ----------

    .. [4] F. Font, G. Roma, and X. Serra, `Freesound technical demo
        <http://mtg.upf.edu/node/2797>`_ 2013, pp. 411-412.
    """
    data_dir = get_data_dir(data_home, FREESOUND_DIR)
    data_path = get_quantized_data_path(data_dir, codebook_size)
    try:
        acoustic = pd.read_pickle(data_path)
    except IOError:
        raise IOError('File does not exists. Make sure the relevant codebook-'
                      'encoded data file `cb_{}_sparse.pkl` is available at '
                      '`data_home/freesound`. You can download the Freesound '
                      'dataset from {}.'.format(codebook_size, FS_DL_PAGE))
        # TODO: need to download raw mfcc files
        # mfccs_path = join(data_dir, 'mfccs')
        # mfccs = _get_freesound_frame_data(mfccs_path)
        # TODO: append derivatives
        # acoustic = quantize_mfccs(mfccs, n_clusters=codebook_size)
        # acoustic.to_pickle(file_path)

    tags_path = join(data_dir, TAGS_FILE)
    sounds_path = join(data_dir, SOUNDS)

    try:
        preprocessed_tags = pd.read_csv(tags_path, index_col=0)
        tags = preprocessed_tags.tags.str.split()
    except IOError:
        tags = preprocess_tags(sounds_path, threshold=0.01)
        tags.to_csv(tags_path)
        tags = tags.tags.str.split()

    df = acoustic.to_dense().join(tags, how='inner')

    Y = df.tags
    X = df.drop('tags', axis=1)

    return X, Y


def _get_freesound_frame_data(path):
    f_names = listdir(path)
    mfccs_dfs = [pd.read_pickle(join(path, f_name)) for f_name in f_names]
    return pd.concat(mfccs_dfs)


def load_freesound_queries(Y_train, data_home=None):
    data_dir = get_data_dir(data_home, FREESOUND_DIR)
    preprocessed_queries_path = join(data_dir, PREPROCESSED_QUERIES)
    raw_queries_path = join(data_dir, QUERIES)

    try:
        queries = pd.read_csv(preprocessed_queries_path, encoding='utf-8')
        queries = queries.dropna()
        queries.q = queries.q.str.split()
    except IOError:
        queries = preprocess_queries(raw_queries_path)
        queries.to_csv(preprocessed_queries_path, index=False,
                       encoding='utf-8')

    return get_relevant_queries(Y_train, queries)
