import logging
import sys

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    RobustScaler,
    StandardScaler,
)
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import is_multilabel

from .loreta import LoretaWARP
from .pamir import PAMIR

from .evaluation import Evaluator
from .utils import SEED
from .datasets import (
    fetch_cal500,
    fetch_cal10k,
    load_freesound,
    load_freesound_queries,
)
from .utils import (
    inverse_document_frequency,
    make_relevance_matrix,
    query_weights,
    standardize,
)

LETOR = ['pamir', 'loreta']


def cv(dataset, codebook_size, multi_word_queries=False, threshold=1,
       n_folds=3, method='loreta', **kwargs):
    """Perform cross-validation

    This function performs cross-validation of the retrieval methods on
    different datasets.

    Parameters
    ----------
    dataset : str, 'cal500', 'cal10k', or 'freesound'
        The dataset on which the retrieval method should be evaluated

    codebook_size : int
        The codebook size the dataset should be encoded with. The data loading
        utility :func:`cbar.datasets.fetch_cal500` contains more information
        about the codebook representation of sounds.

    multi_word_queries : bool, default: ``False``
        If the retrieval method should be evaluated with multi-word queries.
        Only relevant when ``dataset == 'freesound'``.

    threshold : int, default: 1
        Only queries with relevant examples in X_train and X_test >= threshold
        are evaluated.

    n_folds : int, default: 3
        The number of folds used. Only applies to the CAL500 and Freesound
        dataset. The CAL10k dataset has 5 pre-defined folds.

    method : str, 'loreta', 'pamir', or 'random-forest', default: 'loreta'
        The retrieval method to be evaluated.

    kwargs :  key-value pairs
        Additionaly keyword arguments are passed to the retrieval methods.
    """
    if dataset in ['cal500', 'freesound']:
        cross_validate(dataset, codebook_size, multi_word_queries,
                       threshold, n_folds, method, **kwargs)
    elif dataset == 'cal10k':
        cross_validate_cal10k(codebook_size, threshold, method, **kwargs)
    else:
        raise ValueError('This dataset does not exist.')


def cross_validate(dataset, codebook_size, mwq, threshold, n_folds,
                   retrieval_method, **kwargs):
    logging.info('Running CV with {} folds ...'.format(n_folds))

    X, Y = load_dataset(dataset, codebook_size)
    n_samples, n_features = X.shape
    kf = KFold(n_samples, n_folds=n_folds, shuffle=True, random_state=SEED)
    evaluator = Evaluator()

    for idx, (train_idx, test_idx) in enumerate(kf):
        logging.info('Validating fold {} ...'.format(idx))

        (X_train, X_test,
         Y_train, Y_test,
         Q_vec, weights) = index_train_test_split(X, Y, train_idx, test_idx,
                                                  threshold=threshold,
                                                  multi_word_queries=mwq)
        params = validate_fold(X_train, X_test,
                               Y_train, Y_test,
                               Q_vec, weights,
                               evaluator,
                               retrieval_method,
                               **kwargs)

    evaluator.to_json(dataset, retrieval_method, codebook_size, params)


def cross_validate_cal10k(cb_size, threshold, retrieval_method, **kwargs):
    evaluator = Evaluator()

    for i in xrange(1, 6):
        X_train, X_test, Y_train, Y_test = fetch_cal10k(fold=i,
                                                        codebook_size=cb_size)
        (X_train, X_test,
         Y_train, Y_test,
         Q_vec, weights) = dataset_for_train_test_split(X_train, X_test,
                                                        Y_train, Y_test,
                                                        threshold=10)
        params = validate_fold(X_train, X_test,
                               Y_train, Y_test,
                               Q_vec, weights,
                               evaluator,
                               retrieval_method,
                               **kwargs)

    evaluator.to_json('cal10k', retrieval_method, cb_size, params)


def validate_fold(X_train, X_test, Y_train, Y_test, Q_vec, weights,
                  evaluator, retrieval_method, **kwargs):
    """Perform validation on one fold of the data

    This function evaluates a retrieval method on one split of a
    dataset.

    Parameters
    ----------
    X_train : pd.DataFrame, shape = [n_train_samples, codebook_size]
        Training data.

    X_test : pd.DataFrame, shape = [n_test_samples, codebook_size]
        Test data.

    Y_train : pd.DataFrame, shape = [n_train_samples, n_classes]
        Training tags.

    Y_train : pd.DataFrame, shape = [n_test_samples, n_classes]
        Test tags.

    Q_vec : array-like, shape = [n_queries, n_classes]
        The queries to evaluate

    weights : array-like, shape = [n_queries]
        Ouery weights. Multi-word queries can be weighted to reflect importance
        to users.

    evaluator : object
        An instance of :class:`cbar.evaluation.Evaluator`.

    retrieval_method: str, 'loreta', 'pamir', or 'random-forest'
        The retrieval to be evaluated.

    kwargs:  key-value pairs
        Additionaly keyword arguments are passed to the retrieval methods.

    Returns
    -------
    params: dict
        The ``retrieval_method``'s parameters used for the evaluation
    """
    if retrieval_method in LETOR:
        method = dict(pamir=PAMIR, loreta=LoretaWARP).get(retrieval_method)
        letor = method(**kwargs)
        letor.fit(X_train, Y_train, Q_vec, X_test, Y_test)
        Y_score = letor.predict(Q_vec, X_test)
        params = letor.get_params()
    elif retrieval_method == 'random-forest':
        rf = RandomForestClassifier(class_weight='balanced', **kwargs)
        clf = OneVsRestClassifier(rf, n_jobs=-1)
        clf.fit(X_train, Y_train)
        model_score = standardize(clf.predict_proba(X_test))
        Y_score = Q_vec.dot(model_score.T)
        params = clf.estimator.get_params()
    else:
        raise ValueError('Unknown retrieval method.')

    n_relevant = make_relevance_matrix(Q_vec, Y_train).sum(axis=1)
    evaluator.eval(Q_vec, weights, Y_score, Y_test, n_relevant)

    return params


def load_dataset(ds, cb_size):
    load = dict(freesound=load_freesound, cal500=fetch_cal500).get(ds)
    return load(codebook_size=cb_size)


def make_small_dataset(X, Y, total_size=5000, train_size=0.8, threshold=5,
                       multi_word_queries=False):
    X_small, _, Y_small, _ = train_test_split(X.values, Y.values,
                                              train_size=total_size,
                                              random_state=42)
    return train_test_split_plus(X_small, Y_small,
                                 train_size, threshold, multi_word_queries)


def train_test_split_plus(X, Y, train_size=0.8, threshold=1,
                          multi_word_queries=False, scaler='standard'):
    X = np.asarray(X)
    Y = np.asarray(Y)

    (X_train, X_test,
     Y_train, Y_test) = train_test_split(X, Y,
                                         train_size=train_size,
                                         random_state=SEED)

    return dataset_for_train_test_split(X_train, X_test, Y_train, Y_test,
                                        threshold=threshold,
                                        multi_word_queries=multi_word_queries,
                                        scaler=scaler)


def index_train_test_split(X, Y, train_idx, test_idx,
                           threshold=0, multi_word_queries=False,
                           scaler='standard'):
    X = np.asarray(X)
    Y = np.asarray(Y)

    return dataset_for_train_test_split(X[train_idx], X[test_idx],
                                        Y[train_idx], Y[test_idx],
                                        threshold=threshold,
                                        multi_word_queries=multi_word_queries,
                                        scaler=scaler)


def dataset_for_train_test_split(X_train, X_test, Y_train, Y_test, threshold=1,
                                 multi_word_queries=False, scaler='standard'):
    """Make dataset from a train-test-split

    This function scales the input data und generates queries and query-weights
    from the training set vocabulary.

    Parameters
    ----------
    X_train : array-like, shape = [n_train_samples, n_features]
        Training set data

    X_test : array-like, shape = [n_test_samples, n_features]
        Test set data.

    Y_train : array-like, shape = [n_train_samples, n_classes]
        Training set labels.

    Y_test : array-like, shape = [n_test_samples, n_classes]
        Test set labels.

    threshold : int, default: 1
        The threshold ...

    multi_word_queries : bool, default: ``False``
        Generate multi-word queries from real-world user-queries for the
        Freesound dataset if set to ``True``. Ultimately calls
        :func:`cbar.preprocess.get_relevant_queries`

    scaler : str, 'standard' or 'robust', or None
        Use either :func:`sklearn.preprocessing.StandardScaler` or
        :func:`sklearn.preprocessing.RobustScaler` to scale the input data

    Returns
    -------
    X_train : array-like, shape = [n_train_samples, n_features]
        The scaled training data.

    X_test : array-like, shape = [n_test_samples, n_features]
        The scaled test data.

    Y_train_bin : array-like, shape = [n_train_samples, n_classes]
        The training labels in binary indicator format.

    Y_test_bin : array-like, shape = [n_test_samples, n_classes]
        The test labels in binary indicator format.

    Q_vec : array-like, shape = [n_queries, n_classes]
        The query vectors to evaluate

    weights : array-like, shape = [n_queries]
        The weights used to weight the queries during evaluation. For one-word
        queries the weight for each query is the same. For multi-word queries
        the counts from the aggregrated query-log of user-queries are used
        to weight the queries accordingly.
    """
    if scaler:
        scale = dict(standard=StandardScaler(),
                     robust=RobustScaler()).get(scaler)
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)

    mlb = MultiLabelBinarizer()

    if is_multilabel(Y_train):
        Y_train_bin = Y_train
        Y_test_bin = Y_test
    else:
        mlb.fit(np.append(Y_train, Y_test))
        Y_train_bin = mlb.transform(Y_train)
        Y_test_bin = mlb.transform(Y_test)

    n_classes = Y_train_bin.shape[1]

    if multi_word_queries:
        Q = load_freesound_queries(Y_train)
        Q_bin = mlb.transform(Q.q)
    else:
        Q_bin = np.eye(n_classes)

    Y_train_rel = make_relevance_matrix(Q_bin, Y_train_bin)
    Y_test_rel = make_relevance_matrix(Q_bin, Y_test_bin)

    # only keep queries that have at least X relevant sounds in train and test
    mask = np.logical_and(Y_train_rel.sum(axis=1) >= threshold,
                          Y_test_rel.sum(axis=1) >= threshold)
    Q_bin_final = Q_bin[mask]

    if multi_word_queries:
        Q_reduced = Q[mask]
        weights = Q_reduced.cnt / Q_reduced.cnt.sum()
    else:
        weights = np.ones(Q_bin_final.shape[0]) / float(Q_bin_final.shape[0])

    idf = inverse_document_frequency(Y_train_bin)
    Q_vec = query_weights(Q_bin_final, idf)

    return (X_train, X_test,
            Y_train_bin, Y_test_bin,
            Q_vec, weights)


fmt = '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
logFormatter = logging.Formatter(fmt)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('default.log')
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)
rootLogger.addHandler(consoleHandler)
