import logging
import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score

from .datasets import fetch_cal500
from .evaluation import ranking_precision_score
from .utils import make_relevance_matrix


class PAMIR(BaseEstimator):
    """Passive Agressive Model for Image Retrieval

    Parameters
    ----------
    max_iter : int
        The maximum number of iterations.
    C : float
        The regularization constant.
    valid_interval : int
        The interval at which a validation of the current model state is
        performed.
    max_dips : int
        The maximum number of dips the algorithm is allowed to suffer.

    Attributes
    ----------
    W : array, shape = [n_classes, n_features]
        The parameter matrix.

    References
    ----------

    .. [1] Grangier, D. and Bengio, S., 2008. `A discriminative kernel-based
        approach to rank images from text queries.
        <https://infoscience.epfl.ch/record/146417/files/grangier-rr07-38.pdf>`_
        IEEE transactions on pattern analysis and machine intelligence, 30(8),
        pp.1371-1384.

    .. [2] Chechik, G., Ie, E., Rehn, M., Bengio, S. and Lyon, D., 2008,
        October. `Large-scale content-based audio retrieval from text queries.
        <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33429.pdf>`_
        In Proceedings of the 1st ACM international conference on Multimedia
        information retrieval (pp. 105-112). ACM.


    Examples
    --------

    TBD

      >>> clf = PAMIR()
      >>> clf.fit(...)

    """
    def __init__(self, max_iter=100000, C=1.0, valid_interval=10000,
                 max_dips=20, verbose=True):
        self.max_iter = max_iter
        self.C = C
        self.valid_interval = valid_interval
        self.max_dips = max_dips
        self.verbose = verbose
        self._reset_state()

    def fit(self, X, Y, Q, X_val=None, Y_val=None, weights=None):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        Y : array-like, shape = [n_samples, n_classes]
            Training labels
        Q : array-like, shape = [n_queries, n_classes]
            Training queries
        X_val : array-like, shape = [n_samples, n_features], optional
            Validation data
        Y_val : array-like, shape = [n_samples, n_classes], optional
            Validation labels
        weights : array-like, shape = [n_queries], optional
            Query weights

        Returns
        ----------
        self : PAMIR instance
            The fitted model.
        """
        self._reset_state()

        return self.fit_partial(X=X,
                                Y=Y,
                                Q=Q,
                                X_val=X_val,
                                Y_val=Y_val,
                                weights=weights)

    def _reset_state(self):
        self.W = None
        self.best_W = None
        self.best_ap = 0
        self.dip = 0

    def fit_partial(self, X, Y, Q, X_val=None, Y_val=None, weights=None):
        """
        Fit the model. Repeated calls to this method will cause training
        to resume from the current model state.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        Y : array-like, shape = [n_samples, n_classes]
            Training labels
        Q : array-like, shape = [n_queries, n_classes]
            Training queries
        X_val : array-like, shape = [n_samples, n_features], optional
            Validation data
        Y_val : array-like, shape = [n_samples, n_classes], optional
            Validation labels
        weights : array-like, shape = [n_queries], optional
            Query weights

        Returns
        ----------
        self : PAMIR instance
            The fitted model.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X_val is None and Y_val is None:
            X_val = X
            Y_val = Y
        else:
            X_val = np.asarray(X_val)
            Y_val = np.asarray(Y_val)

        n_samples, n_features = X.shape
        n_queries, n_classes = Q.shape

        # Initialize parameter matrix W only for the first call to fit_partial
        if self.W is None:
            self._init_W(n=n_classes, m=n_features)

        self._relevance_matrix = make_relevance_matrix(Q, Y)

        if weights is None:
            weights = np.ones(n_queries) / n_queries

        if Y_val is None:
            self.Y_true = self._relevance_matrix
        else:
            self.Y_true = make_relevance_matrix(Q, Y_val)
            self.at_least_one_relevant = self.Y_true.any(axis=1)
            self.Y_true = self.Y_true[self.at_least_one_relevant]

        self._reset_inner_loss()

        for it in xrange(self.max_iter):
            if it % self.valid_interval == 0:
                p10, ap = self._validate(Q, weights, X_val, Y_val)
                if ap > self.best_ap:
                    self.best_ap = ap
                    self.best_W = self.W
                    self.dip = 0
                else:
                    self.dip += 1

                if self.dip == self.max_dips:
                    self.W = self.best_W
                    logging.warn('max_dips reached, '
                                 'stopped at {} iterations.'.format(it))
                    break
                if self.verbose:
                    loss = sum(self.inner_loss) / float(self.valid_interval)
                    log = ('iter: {:8}, P10: {:.3f}, AP: {:.3f}, loss: {:.3f}')
                    print log.format(it, p10, ap, loss)
                self._reset_inner_loss()

            query, relevant, irrelvant = self._sample_triplet(n_queries, Q, X)

            V = np.outer(query, relevant - irrelvant)
            loss = self._hinge_loss(query, relevant, irrelvant)
            self.inner_loss.append(loss)
            tau = min(self.C, loss / linalg.norm(V))
            self.W = self.W + tau * V

        return self

    def predict(self, Q, X_test):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        Y : array-like, shape = [n_samples, n_classes]
            Training labels

        Returns
        ----------
        self : object
            This model
        """
        return Q.dot(self.W).dot(X_test.T)

    def _hinge_loss(self, query, rel, irr):
        return max(0, 1 - self.predict(query, rel) + self.predict(query, irr))

    def _init_W(self, n, m):
        """
        Initialize the parameter matrix :math:`W \in \mathbb{R}^{n \times m}`

        Parameters
        ----------
        n : int
            First dimension for W. Number of classes.
        m : int
            Second dimension for W. Number of features.
        """
        # self.W = np.zeros((self.n_classes, self.n_features))
        self.W = np.random.randn(n, m)

    def _reset_inner_loss(self):
        self.inner_loss = []

    def _sample_triplet(self, n_queries, Q, X):
        qid = np.random.choice(n_queries)
        rel = np.random.choice(self._relevance_matrix[qid].nonzero()[0])
        irr = np.random.choice(np.where(self._relevance_matrix[qid] == 0)[0])
        return Q[qid], X[rel], X[irr]

    def _validate(self, Q_val, weights, X_val, Y_val):
        p10, ap = [], []
        Y_score = self.predict(Q_val, X_val)
        Y_score = Y_score[self.at_least_one_relevant]

        for x in xrange(self.Y_true.shape[0]):
            p10.append(ranking_precision_score(self.Y_true[x],
                                               Y_score[x]) * weights[x])
            ap.append(average_precision_score(self.Y_true[x],
                                              Y_score[x]) * weights[x])
        return np.sum(p10), np.sum(ap)

    def __repr__(self):
        return ('{}.{}(max_iter={}, C={}, valid_interval={}, '
                'max_dips={})').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.max_iter,
            self.C,
            self.valid_interval,
            self.max_dips,
        )


def main():
    from .cross_validation import train_test_split_plus

    X, Y = fetch_cal500()

    (X_train, X_test,
     Y_train, Y_test,
     Q_vec, weights) = train_test_split_plus(X, Y)

    clf = PAMIR(max_iter=100000, valid_interval=1000)
    clf.fit(X_train, Y_train, Q_vec, X_test, Y_test, weights)


if __name__ == '__main__':
    main()
