import json
import time
from collections import defaultdict
from os.path import join

import numpy as np
from sklearn.metrics import average_precision_score

from .utils import make_relevance_matrix
from .settings import RESULTS_DIR


class Evaluator(object):
    """The ``Evaluator`` evaluates a retrieval method, collects the perfromance
    measures, and keeps values of multiple runs (for example in k-fold
    cross-validation).
    """
    def __init__(self):
        self.mean_ap = []
        self.prec_at = defaultdict(list)
        self.rel_prec = defaultdict(list)

    def eval(self, queries, weights, Y_score, Y_test, n_relevant):
        """
        Parameters
        ----------
        queries : array-like, shape = [n_queries, n_classes]
            The queries to evaluate

        weights : int, default: 1
            Ouery weights. Multi-word queries can be weighted to reflect
            importance to users.

        Y_score : array-like, shape = [n_queries, n_classes]
            Scores of queries and sounds.

        Y_test : array-like, shape = [n_samples, n_classes]
            Test set tags associated with each test set song in
            binary indicator format.

        n_relevant : array-like, shape = [n_queries]
            The number of relevant sounds in X_train for each query.
        """
        # delete rows which have no relevant sound
        Y_true = make_relevance_matrix(queries, Y_test)
        at_least_one_relevant = Y_true.any(axis=1)
        Y_true = Y_true[at_least_one_relevant]
        Y_score = Y_score[at_least_one_relevant]
        n_relevant = n_relevant[at_least_one_relevant]

        ap = []
        prec = defaultdict(list)

        for x in xrange(Y_true.shape[0]):
            self.rel_prec[n_relevant[x]].append(
                ranking_precision_score(Y_true[x], Y_score[x]))

            for k in xrange(1, 21):
                prec[k].append(ranking_precision_score(
                                  Y_true[x], Y_score[x], k) * weights[x])
            ap.append(average_precision_score(Y_true[x],
                                              Y_score[x]) * weights[x])
        self.mean_ap.append(np.sum(ap))

        for k, v in prec.iteritems():
            self.prec_at[k].append(np.sum(v))

    def to_json(self, dataset, method, codebook_size, params):
        """
        Write the retrieval performance results to a file.

        Parameters
        ----------
        dataset : str
            The name of the evaluated dataset.

        method : str
            The name of the evaluated retrieval method.

        codebook_size : int
            The codebook size the dataset is encoded with.

        params: dict
            The ``method``'s parameters used during the evaluation.
        """
        self._to_json(join(RESULTS_DIR, '{}_precision.json'.format(dataset)),
                      method, params, codebook_size, self.prec_at)

        self._to_json(join(RESULTS_DIR, '{}_mean_ap.json'.format(dataset)),
                      method, params, codebook_size, self.mean_ap)

        self._to_json(join(RESULTS_DIR, '{}_prec_at_rel.json'.format(dataset)),
                      method, params, codebook_size, self.rel_prec)

    def _check_exists(self, filename):
        try:
            open(filename)
        except IOError:
            with open(filename, 'w+') as f:
                json.dump(dict(), f)

    def _stats(self, name, params, codebook_size, precision):
        stats = dict(name=name,
                     params=params,
                     codebook_size=codebook_size,
                     precision=precision)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return {timestr: stats}

    def _to_json(self, filename, name, params, cb_size, precision):
        self._check_exists(filename)

        with open(filename, 'r+') as f:
            dic = json.load(f)
            dic.update(self._stats(name, params, cb_size, precision))

        with open(filename, 'w+') as f:
            json.dump(dic, f, indent=4)


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    Returns
    -------
    precision@k : float
        Precision at rank k.
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = y_score.argsort()[::-1]
    y_true = y_true[order[:k]]
    n_relevant = (y_true == pos_label).sum()
    # Divide by min(n_pos, k) so that the best achievable score is always 1.0
    return float(n_relevant) / min(n_pos, k)
