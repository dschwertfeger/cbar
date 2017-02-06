import argparse
import json

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.signal import lfilter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import MultiLabelBinarizer

from .utils import SEED
from .utils import make_relevance_matrix

BLACKLIST = ['multisample', 'sound', 'sounds', 'Sound', 'compmusic',
             'velocity', 'sample', 'ittm', 'samples', 'stereo', 'good-sounds',
             'pabloproject', 'bpm', 'sound-design', 'tone', 'request',
             'OWI', 'free', 'note', 'mix', 'porn-core',
             'Finnish-Broadcasting-Company', 'freesound', 'single-note',
             'iitm', 'kingkorg', 'microphon', 'neumann', 'song']

stemmer = PorterStemmer()


def make_vocabulary(df, threshold):
    thresh = df.shape[0] / 100 * threshold
    tags = pd.DataFrame(df.tags.apply(pd.Series).stack())
    tags.reset_index(level=1, drop=True, inplace=True)
    tags.rename(columns={0: 'tag'}, inplace=True)
    tag_counts = tags.groupby('tag').size()
    tags = remove_number_tags(tag_counts)
    tags = tags[tags > thresh]
    tags = tags[tags.index.str.len() > 2]
    return tags


def remove_number_tags(series):
    return series[~series.index.str.contains(r'\d')]


def tokenize(tags):
    # http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    return [item for sub in [x.split('-') for x in tags] for item in sub]


def contains_number(tag):
    return any(char.isdigit() for char in tag)


def too_short(tag):
    return len(tag) < 3


def _filter(tags, vocabulary):
    return [tag for tag in tags if tag in vocabulary and tag not in BLACKLIST]


def preprocess(tags):
    tags = tokenize(tags)
    stops = set(stopwords.words('english'))
    tags = [stemmer.stem(tag.lower())
            for tag in tags if tag not in stops and not too_short(tag)]
    return list(set(tags))


def delta(data, width=9, order=1, axis=-1, trim=True):
    """copied from librosa"""
    data = np.atleast_1d(data)

    if width < 3 or np.mod(width, 2) != 1:
        raise Exception('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        raise Exception('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = np.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= np.sum(np.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = np.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = lfilter(window, 1, delta_x, axis=axis)

    # Cut back to the original shape of the input data
    if trim:
        idx = [slice(None)] * delta_x.ndim
        idx[axis] = slice(- half_length - data.shape[axis], - half_length)
        delta_x = delta_x[idx]

    return delta_x


def quantize_mfccs(df, n_clusters=2048):
    scaled = df.apply(scale)
    mbk = MiniBatchKMeans(n_clusters=n_clusters,
                          batch_size=n_clusters * 20,
                          max_no_improvement=20,
                          reassignment_ratio=.0001,
                          random_state=SEED,
                          verbose=True)
    mbk.fit(scaled)
    scaled['label'] = mbk.labels_.tolist()
    sounds = scaled.groupby(level=0)
    acoustic = pd.DataFrame({_id: s.groupby('label').size()
                            for _id, s in sounds}).transpose()
    return acoustic.fillna(0).to_sparse(fill_value=0)


def preprocess_tags(filename, threshold):
    # load tags
    sounds = json.load(open(filename))
    tags = {sound['id']: [sound['tags']] for sound in sounds}
    df = pd.DataFrame(tags).transpose().rename(columns={0: 'tags'})
    # preprocess tags
    vocabulary = make_vocabulary(df, threshold)
    df.tags = df.tags.map(lambda x: preprocess(_filter(x, vocabulary.index)))
    df = df[df.tags.map(lambda x: len(x) > 0)]
    df.tags = df.tags.str.join(' ')
    return df


def preprocess_queries(path):
    queries = pd.read_csv(path, delimiter=';', encoding='utf-8').dropna()
    queries.q = queries.q.str.split()
    queries.q = queries.q.map(preprocess)
    # remove queries that have become empty
    queries = queries[queries.q.str.len() > 0]
    queries.q = queries.q.str.join(' ')
    return queries


def get_relevant_queries(Y_train, queries):
    """Get relevant queries according to the training set vocabulary.

    Parameters
    ----------
    Y_train : array, shape = [n_samples]
        The traning data tags as list of lists

    queries : pd.DataFrame, shape = [n_queries, 2]
        The preprocessed queries (``queries.q``) and their counts
        (``queries.cnt`) inidicating how many times a query was posted.

    Returns
    -------
    unique_queries : pd.DataFrame, shape = [n_unique_queries, 2]
        The unique queries and their counts (inidicating how many times a
        query was posted). The counts are used to weight queries during
        evaluation.
    """
    mlb = MultiLabelBinarizer()
    Y_train_bin = mlb.fit_transform(Y_train)
    queries.q = queries.q.map(lambda q: _filter(q, mlb.classes_))
    # only keep max. 3-word queries
    queries = queries[np.logical_and(queries.q.str.len() > 0,
                                     queries.q.str.len() < 4)]
    queries['q_str'] = queries.q.str.join(' ')
    unique_queries = queries.groupby('q_str').sum()
    unique_queries['q'] = unique_queries.index.str.split()
    # only keep queries with at least five positive examples in X_train
    bin_queries = mlb.transform(unique_queries.q)
    rel = make_relevance_matrix(bin_queries, Y_train_bin)
    unique_queries = unique_queries[rel.sum(axis=1) > 4]
    return unique_queries
