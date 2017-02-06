from os import path, makedirs, getcwd

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.externals import joblib
from sklearn.utils.multiclass import is_multilabel

# CONSTS

EPS = np.finfo(np.float32).eps
SEED = 42


# HELPERS

def normalize_codebook_counts(X):
    tf = np.asarray(X)
    rc = (tf != 0).sum(axis=0) / float(tf.shape[0])
    idf = -np.log(rc)
    return tf * idf / np.sqrt(np.square(tf * idf).sum())


def binarize(query, mlb):
    """Transform a single query into binary representation

    Parameters
    ----------
    query : ndarray, shape = [n_samples, n_classes]
        The tags.
    n_samples : int
        The number of samples in the training set.

    Returns
    -------
    bin_query_vector : ndarray, shape = [n_samples]
        Binary query vector.
    """
    return mlb.transform([query]).ravel()


def binarize_targets(mlb, Y):
    # NOTE: Not used anywhere
    if is_multilabel(Y):
        mlb.fit([Y.columns])
        Y_bin = Y.values
    else:
        mlb.fit(Y)
        Y_bin = mlb.transform(Y)
    return mlb, Y_bin


def ensure_dimension(y_true):
    # NOTE: Not used anywhere
    """Transform y_true

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, len(query)]
        True binary labels in binary label indicators. Shape depends on the
        query terms.

    Returns
    -------
    y_true : array, shape = [n_samples]
        True binary labels in binary label indicators.
    """
    return y_true.all(axis=1) if y_true.ndim > 1 else y_true


def inverse_document_frequency(Y):
    """Compute the inverse document frequency for each tag in the vocabulary.

    IDF measures how important a term is. IDF are also the Israeli Defense
    Forces but that is another story.

    Parameters
    ----------
    Y : ndarray, shape = [n_samples, n_classes]
        The tags in binary form.

    Returns
    -------
    idf : ndarray, shape = [n_samples]
        The inverse document frequency of all tags in the vocabulary.
    """
    n_samples = Y.shape[0]
    n_samples_with_tag = get_tag_counts(Y)
    df = n_samples_with_tag / float(n_samples)
    df[df == 0] = EPS
    return -np.log(df)


def get_previews_df(sounds_path):
    """Get DataFrame containing preview-urls for sounds.

    Parameters
    ----------
    sounds_path : str
        Path of the `sounds.json` file

    Returns
    -------
    previews : pd.DataFrame
        The preview URLs for sounds
    """
    sounds = json.load(open(sounds_path))
    previews = {sound['id']: [sound['previews']['preview-hq-ogg']]
                for sound in sounds}
    previews = pd.DataFrame(previews).transpose()
    return previews.rename(columns={0: 'url'})


def get_tag_counts(Y):
    """Compute the total number of documents that contain term t for all terms
    in the vocabulary.

    Parameters
    ----------
    Y : ndarray, shape = [n_samples, n_classes]
        The tags as multilabel array (or list of lists.)

    Returns
    -------
    tag_counts : ndarray, shape = [n_samples]
        The inverse document frequency of all tags in the vocabulary.
    """
    if is_multilabel(Y):
        return Y.sum(axis=0)
    else:
        return pd.DataFrame(Y.apply(pd.Series).stack()).groupby(0).size()


def query_weights(bin_query_vec, idf):
    """Represent a query as a vector of query weights.

    Parameters
    ----------
    bin_query_vec : ndarray, shape = [n_classes]
        A binary query vector.
        TODO: update
            - Could also be a query matrix not just a single vector!
    idf: ndarray, shape = [n_classes]
        Inverse document frequency

    Returns
    -------
    query_weights : ndarray, shape = [n_samples]
        Query vector weighted with the normalized inverse document
        frequency weighting scheme.
    """
    idf = np.asarray(idf)
    return bin_query_vec * idf / np.sqrt(np.sum(np.square(idf)))


def query_multinomial(query, mlb):
    # NOTE: not used anywhere
    # TODO-dsc: weight the query terms
    bin_query_vector = mlb.transform([query]).ravel()
    query_vec = bin_query_vector.astype(np.float64)
    query_vec[query_vec == 0] = EPS
    return query_vec / query_vec.sum()


def semantic_multinomial(proba):
    # NOTE: not used anywhere
    return np.divide(proba, proba.sum(axis=1, keepdims=True))


def kl_divergence(query, sound):
    # NOTE: not used anywhere
    return [stats.entropy(query, x) for x in sound]


def score_query(query_vector, model_score):
    """Score indicates the permutation of the dataset for ranking

    Parameters
    ----------
    query_vector : ndarray, shape = [n_classes]
        TBD

    model_score: ndarray, shape = [n_classes]
        TBD

    Returns
    -------
    scored_query : ndarray, shape = [n_samples]
        TBD
    """
    return np.sum(query_vector * model_score, axis=1)


def standardize(proba):
    return (proba - proba.mean(axis=0, keepdims=True)) / proba.std(axis=0)


def top_k_idxs(arr, k):
    return (-arr).argsort()[:k]


def make_relevance_matrix(Q, Y):
    """Relevance matrix of queries and sounds.

    Compute a matrix that encodes the interactions of queries and sounds.
    In other words, which sounds are positively associated with a query (1)
    and which sounds are not (0).

    A sound is considered relevant to a query if its tags cover *all* the
    query terms.

    For an example query 'bass drum', sounds tagged with
    `['bass', 'drum', 'kick']` or `['deep', 'bass', 'drum']` are relevant but
    sounds tagged with only `[bass]`, `[drum]`, or `['bass' , 'guitar']` are
    not.

    Parameters
    ----------
    Q : array-like, shape = [n_samples]
        Queries in binary label indicator format.
    Y : array-like, shape = [n_samples]
        Tags of sounds in binary label indicator format.

    Returns
    -------
    relevance_matrix : array-like, shape = [n_queries, n_samples]
        The relevance matrix
    """
    n_queries, _ = Q.shape
    n_samples, n_classes = Y.shape
    relevance_matrix = np.empty((n_queries, n_samples), dtype='b')
    tags = [row.nonzero()[0] for row in Q]
    for idx, t in enumerate(tags):
        relevance_matrix[idx] = Y[:, t].all(axis=1)
    return relevance_matrix


# IO

def _model_file_path(name):
    model_path = path.join(getcwd(), 'models')
    return path.join(model_path, name)


def _absolute_model_file_path_for(name):
    file_path = _model_file_path(name)
    file_name = '{}.pkl'.format(name)
    return path.join(file_path, file_name)


def save_model(clf, name):
    file_path = _model_file_path(name)
    file_name = _absolute_model_file_path_for(name)
    if not path.exists(file_path):
        makedirs(file_path)
    joblib.dump(clf, file_name)


def load_model(name):
    try:
        return joblib.load(_absolute_model_file_path_for(name))
    except Exception, e:
        raise e
