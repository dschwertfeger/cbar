from collections import defaultdict
from zipfile import ZipFile

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
)
from .base import get_data, get_data_dir, get_quantized_data_path

from ..preprocess import delta, quantize_mfccs


CAL10K_DIR = 'cal10k'
CAL10K_ZIP = 'cal10k.zip'
DATA_URL = ('http://calab1.ucsd.edu/~datasets/cal10k/cal10kdata/'
            'cal10k_mfcc_22050Hz_win2048_hop1024.zip')
TAGS_ZIP = 'tags.zip'
TAGS_URL = ('http://acsweb.ucsd.edu/~yvaizman/metadata/'
            'cal10k_tagsFrom30SongsOrMore_5folds.zip')


def fetch_cal10k(data_home=None, download_if_missing=True,
                 codebook_size=512, fold=1):
    r"""Loader for the CAL10k dataset [2]_.

    This dataset consists of 10,870 western pop songs, performed by 4,597
    unique artists. Each song is weakly annotated with 2 to 25 tags from a tag
    vocabulary of 153 genre tags and 475 acoustic tags but only tags
    associated with at least 30 songs are kept for the final tag vocabulary.

    The CAL10k dataset has predefined train-test-splits for a 5-fold
    cross-validation

    .. warning::

        This utility downloads a ~2GB file to your home directory. This might
        take a few minutes, depending on your bandwidth.

    Parameters
    ----------
    data_home : optional
        Specify a download and cache folder for the datasets. By default
        (``None``) all data is stored in subfolders of ``~/cbar_data``.

    download_if_missing: bool, optional
        If ``False``, raise a ``IOError`` if the data is not locally available
        instead of trying to download the data from the source site.
        Defaults to ``True``.

    codebook_size : int, optional
        The codebook size. Defaults to 512.

    fold : int, :math:`\in \{1, ..., 5\}`
        The specific train-test-split to load. Defaults to 1.

    Returns
    -------
    X_train : array-like, shape = [n_train_samples, codebook_size]
        Training set songs. Each row corresponds to a preprocessed song,
        represented as a sparse codebook vector.

    X_test : array-like, shape = [n_test_samples, codebook_size]
        Test set songs. Each row corresponds to a preprocessed song,
        represented as a sparse codebook vector.

    Y_train : array-like, shape = [n_train_samples, 581]
        Training set tags associated with each training set song in
        binary indicator format.

    Y_test : array-like, shape = [n_test_samples, 581]
        Test set tags associated with each test set song in
        binary indicator format.

    Notes
    ------

    The CAL10k dataset is downloaded from UCSD's `Computer Audition
    Laboratory's datasets page <http://calab1.ucsd.edu/~datasets/cal10k/>`_.
    The annotations are the "corrected" annotations from [3]_, downloaded from
    the `CAL10k corrected metadata page
    <http://calab1.ucsd.edu/~datasets/cal10k/>`_.

    The raw dataset consists of the 13 mel-frequency cepstral coefficients
    for each frame of each song. The acoustic data is preprocessed similar to
    the acoustic data in the CAL500 dataset (see notes in
    :func:`cbar.datasets.fetch_cal500`).

    References
    ----------

    .. [2] D. Tingle, Y. E. Kim, and D. Turnbull, `Exploring automatic music
        annotation with acoustically-objective tags.
        <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.9746&rep=
        rep1&type=pdf>`_
        in Proceedings of the international conference on Multimedia
        information retrieval, 2010, pp. 55-62.


    .. [3] Y. Vaizman, B. McFee, and G. Lanckriet, `Codebook-based audio
        feature representation for music information retrieval.
        <http://acsweb.ucsd.edu/~yvaizman/papers/taslp_2014_accepted.pdf>`_
        Audio, Speech, and Language Processing, IEEE/ACM Transactions on, vol.
        22, no. 10, pp. 1483-1493, 2014.
    """

    data_dir = get_data_dir(data_home, CAL10K_DIR)
    data_path = get_data(DATA_URL, data_dir, CAL10K_ZIP, download_if_missing)
    tags_path = get_data(TAGS_URL, data_dir, TAGS_ZIP, download_if_missing)

    file_path = get_quantized_data_path(data_dir, codebook_size)
    le = LabelEncoder()

    with ZipFile(data_path) as data:
        le.fit([_clean(filename) for filename in data.namelist()[1:]])

    try:
        X = pd.read_pickle(file_path)
    except IOError:
        mfccs = _read_cal10k_from_disk(data_path, le)
        X = quantize_mfccs(mfccs, n_clusters=codebook_size)
        X.to_pickle(file_path)

    with ZipFile(tags_path) as tags:
        train = _get_annotations(tags.open('fold_{}_train.tab'.format(fold)))
        test = _get_annotations(tags.open('fold_{}_test.tab'.format(fold)))

    train_ids = le.transform(train.keys())
    test_ids = le.transform(test.keys())

    X_train = X.iloc[train_ids]
    X_test = X.iloc[test_ids]

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train.values())
    Y_test = mlb.transform(test.values())

    return X_train, X_test, Y_train, Y_test


def _read_cal10k_from_disk(data_path, le):
    with ZipFile(data_path) as data:
        songs = {le.transform(_clean(f_name)): pd.read_csv(data.open(f_name),
                                                           header=None)
                 for f_name in data.namelist()[1:]}
        mfccs = pd.concat(songs.values(), keys=songs.keys())
        delta1 = pd.DataFrame(delta(mfccs))
        delta2 = pd.DataFrame(delta(mfccs, order=2))
        mfccs.index.set_names(['song', 'frame'], inplace=True)
        mfccs.reset_index(inplace=True)
        X = pd.concat((mfccs, delta1, delta2), axis=1)
        X.set_index(['song', 'frame'], inplace=True)
    return X


def _get_annotations(fd):
    annotations = defaultdict(list)
    for line in fd.readlines():
        song, label = line.strip('\r\n').strip('\n').split('\t')
        song = song.replace('A Brand New Bag (live)',
                            'A Brand New Bag (Live)')
        annotations[song].append(label)
    return annotations


def _clean(filename):
    return filename.replace('mfcc_2048_1024/', '') \
                   .replace('.mfcc', '') \
                   .replace('%3f', '?') \
                   .replace('%22', '"') \
                   .replace('%25', '%') \
                   .replace('%2a', '*') \
                   .replace('%3e', '>') \
                   .replace('%7c', '|')
