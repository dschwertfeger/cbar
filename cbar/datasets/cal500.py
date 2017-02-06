import tarfile
import pandas as pd

from .base import get_data, get_data_dir, get_quantized_data_path

from ..preprocess import quantize_mfccs

CAL500_DIR = 'cal500'
CAL500_TAR = 'CAL500_DeltaMFCCFeatures.tar.gz'
URL = ('http://calab1.ucsd.edu/~datasets/cal500/cal500data/'
       'CAL500_DeltaMFCCFeatures.tar.gz')


def fetch_cal500(data_home=None, download_if_missing=True, codebook_size=512):
    r"""Loader for the CAL500 dataset [1]_.

    This dataset consists of 502 western pop songs, performed by 499 unique
    artists. Each song is tagged by at least three people using a standard
    survey and a fixed tag vocabulary of 174 *musical concepts*.

    .. warning::

        This utility downloads a ~1GB file to your home directory. This might
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

    Returns
    -------
    X : pd.DataFrame, shape = [502, codebook_size]
        Each row corresponds to a preprocessed song, represented as a sparse
        codebook vector.

    Y : pd.DataFrame, shape = [502, 174]
        Tags associated with each song in binary indicator format.

    Notes
    ------

    The CAL500 dataset is downloaded from UCSD's `Computer Audition
    Laboratory's datasets page <http://calab1.ucsd.edu/~datasets/cal500/>`_.

    The raw dataset consists of about 10,000 39-dimensional features vectors
    per minute of audio content. The feature vectors were created by:

    1.
        Sliding a half-overlapping short-time window of 12 milliseconds over
        each song's waveform data.
    2.
        Extracting the 13 mel-frequency cepstral coefficients.
    3.
        Appending the instantaneous first-order and second-order derivatives.

    Each song is represented by exactly 10,000 randomly subsampled,
    real-valued feature vectors as a *bag-of-frames*
    :math:`\mathcal{X} = \{\vec{x}_1, \ldots, \vec{x}_T\} \in
    \mathbb{R}^{d \times T}`, where :math:`d = 39` and :math:`T = 10000`.

    The *bag-of-frames* features for each song are further preprocessed into
    one *k*-dimensional feature vector with the following procedure:

    1.
        **Encode feature vectors as code vectors.**
        Each feature vector :math:`\vec{x}_t \in \mathbb{R}^d` is encoded as a
        code vector :math:`\vec{c}_t \in \mathbb{R}^k` according to a
        pre-defined codebook :math:`C \in \mathbb{R}^{d \times k}`. The
        intermediate representation for the encoded audio file is
        :math:`\mathcal{X}_{enc} \in \mathbb{R}^{k \times T}`.
    2.
        **Pool code vectors into one compact vector.**
        The encoded frame vectors are pooled together into a single compact
        vector. An audio file :math:`x` can now be represented as a single
        *k*-dimensional vector :math:`\vec{x} \in \mathbb{R}^k`.

    Specifically, the k-means clustering algorithm is used to cluster all
    audio files' frames into ``codebook_size`` clusters in step 1. The
    resulting cluster centers correspond to the codewords in the codebook.
    Accordingly, the encoding step consists of assigning each frame vector
    to its closest cluster center.

    References
    ----------

    .. [1] D. Turnbull, L. Barrington, D. Torres, and G. Lanckriet, `Semantic
        Annotation and Retrieval of Music and Sound Effects.
        <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.2154&rep=
        rep1&type=pdf>`_
        IEEE Transactions on Audio, Speech, and Language Processing, vol. 16,
        no. 2, pp. 467-476, Feb. 2008.
    """
    data_dir = get_data_dir(data_home, CAL500_DIR)
    tar_path = get_data(URL, data_dir, CAL500_TAR, download_if_missing)
    file_path = get_quantized_data_path(data_dir, codebook_size)

    with tarfile.open(tar_path) as tar:

        vocab = pd.read_csv(tar.extractfile('vocab.txt'), header=None)
        Y = pd.read_csv(tar.extractfile('hardAnnotations.txt'),
                        header=None, names=vocab.values.ravel())
        try:
            X = pd.read_pickle(file_path)
        except IOError:
            delta = [member for member in tar
                     if member.name.startswith('delta/')]
            songs = [pd.read_csv(tar.extractfile(song),
                                 header=None, delim_whitespace=True)
                     for song in delta]
            mfccs = pd.concat(songs, keys=range(len(songs)))
            X = quantize_mfccs(mfccs, n_clusters=codebook_size)
            X.to_pickle(file_path)

        return X, Y
