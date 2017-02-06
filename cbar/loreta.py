import logging
import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score

from .datasets import fetch_cal500
from .evaluation import ranking_precision_score
from .utils import EPS
from .utils import make_relevance_matrix


class LoretaWARP(BaseEstimator):
    r"""
    Low Rank Retraction Algorithm with Weighted Approximate-Rank
    Pairwise loss (WARP loss).

    Parameters
    ----------
    max_iter : int
        The maximum number of iterations. One iteration equals one run of
        sampling the triplet :math:`(q, x_q^+, x_q^-)`.

    k : int
        The rank of the parameter matrix
        :math:`W = AB^T \in \mathbb{R}^{n \times m}`
        with :math:`A \in \mathbb{R}^{n \times k}` and
        :math:`B \in \mathbb{R}^{m \times k}`

    n0 : float
        The first step size parameter.

    n1 : float
        The second step size parameter

    rank_thresh : float
        The threshold for early-stopping the sampling procedure. Stops sampling
        after :math:`x_q^- \cdot rank\_thresh` samples have been sampled.

    lambda_ : float
        The regularization constant :math:`\lambda`.

    loss : str, 'warp' or 'auc'
        The loss function.

    max_dips: int
        The maximum number of dips the algorithms is allowed to make

    valid_interval : int
        The interval at which a validation of the current model state is
        performed.

    verbose : bool, optional
        Turns valiation output at each validation interval on or off.


    Attributes
    ----------
    A : array, shape = [n_classes, k]
        Factor :math:`A` of the factored parameter matrix :math:`W` such that
        :math:`W = AB^T` with :math:`A \in \mathbb{R}^{n \times k}`

    B : array, shape = [n_features, k]
        Factor :math:`B` of the factored parameter matrix :math:`W` such that
        :math:`W = AB^T` :math:`B \in \mathbb{R}^{m \times k}`

    A_pinv : array, shape = [k, n_classes]
        The pseudo-inverse of A.

    B_pinv : array, shape = [k, n_features]
        The pseudo-inverse of B.


    Notes
    -----

    This class implements the Low Rank Retraction Algorithm with rank-one
    gradients as described in [1]_. The rank-one gradients allow for an
    efficient, iterative update of the pseudo-inverses based on [2]_ instead of
    two expensive :math:`\mathcal{O}(n^3)` spectral decomposition in every
    iteration to compute the pseudo-inversess.

    Moreover, the algorithm features the *Weighted Approximate-Rank Pairwise*
    loss (WARP loss) [3]_ which employs a clever sampling scheme to speed up
    training time.

    A similar algorithm with the additional constraint that W is PSD was
    developed in [4]_.


    References
    ----------

    .. [1] Shalit, U., Weinshall, D. and Chechik, G., 2012. `Online learning in
       the embedded manifold of low-rank matrices.
       <http://www.jmlr.org/papers/volume13/shalit12a/shalit12a.pdf>`_
       Journal of Machine Learning Research, 13(Feb), pp.429-458.

    .. [2] Meyer, Jr, C.D., 1973. `Generalized inversion of modified matrices.
       <http://epubs.siam.org/doi/pdf/10.1137/0124033>`_
       SIAM Journal on Applied Mathematics, 24(3), pp.315-323.

    .. [3] Weston, J., Bengio, S. and Usunier, N., 2010. `Large scale image
       annotation: learning to rank with joint word-image embeddings.
       <https://research.google.com/pubs/archive/35780.pdf>`_
       Machine learning, 81(1), pp.21-35.

    .. [4] Lim, D. and Lanckriet, G., 2014. `Efficient Learning of Mahalanobis
       Metrics for Ranking.
       <http://www.jmlr.org/proceedings/papers/v32/lim14.pdf>`_
       In Proceedings of The 31st International Conference on Machine Learning
       (pp. 1980-1988).


    """
    def __init__(self, max_iter=100000, k=30, n0=1.0, n1=0.0, rank_thresh=0.1,
                 lambda_=0.1, loss='warp', valid_interval=10000,
                 max_dips=10, verbose=True):
        self.max_iter = max_iter
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.rank_thresh = rank_thresh
        self.lambda_ = lambda_
        self.loss = loss
        self.valid_interval = valid_interval
        self.max_dips = max_dips
        self.verbose = verbose

        self._validate_params()
        self._reset_state()

    def _validate_params(self):
        if not isinstance(self.k, int):
            raise ValueError('k must be an integer')
        if not isinstance(self.verbose, bool):
            raise ValueError('verbose must be either True or False')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be > zero')
        if self.max_dips <= 0:
            raise ValueError('max_dips must be > zero')
        if self.valid_interval <= 0:
            raise ValueError('valid_interval must be > zero')
        if self.lambda_ < 0:
            raise ValueError('lambda_ must be >= zero')
        if not (0.0 <= self.rank_thresh <= 1.0):
            raise ValueError('rank_thresh must be in [0, 1]')
        if self.loss not in ['warp', 'auc']:
            raise ValueError("loss must be either 'warp' or 'auc'")

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
        self : LoretaWARP instance
            The fitted model.
        """
        self._reset_state()

        return self.fit_partial(X=X,
                                Y=Y,
                                Q=Q,
                                X_val=X_val,
                                Y_val=Y_val,
                                weights=weights)

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
        self : LoretaWARP instance
            The fitted model.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X_val is None and Y_val is None:
            X_val = X
            Y_val = Y
        elif X_val is None or Y_val is None:
            raise ValueError('Both X_val and Y_val must be provided')
        else:
            X_val = np.asarray(X_val)
            Y_val = np.asarray(Y_val)

        n_queries, n_classes = Q.shape
        n_samples, n_features = X.shape
        # Initialize factored parameter matrix W only
        # if this is the first call to fit partial
        if not self.initialized:
            self._initialize_W(n_classes, n_features)
            self._init_warp_loss_table(n_samples)
            self._reset_inner_loss()

        if weights is None:
            weights = np.ones(n_queries) / n_queries

        self._relevance_matrix = make_relevance_matrix(Q, Y)
        self.Y_true = make_relevance_matrix(Q, Y_val)

        for it in xrange(self.max_iter):
            stepsize = self._stepsize(it)

            if it % self.valid_interval == 0:
                p10, ap = self._validate(Q, weights, X_val)

                if ap > self.best_ap and it > 2 * self.valid_interval - 1:
                    self.best_ap = ap
                    self.best_A = self.A
                    self.best_B = self.B
                    self.n_dips = 0
                else:
                    self.n_dips += 1

                if self.n_dips == self.max_dips:
                    self.A = self.best_A
                    self.B = self.best_B
                    logging.warn('max_dips reached, '
                                 'stopped at {} iterations.'.format(it))
                    break

                loss = sum(self.inner_loss) / float(self.valid_interval)

                if self.verbose:
                    log = ('iter: {:8}, stepsize: {:8.3f}, P10: {:.3f}, '
                           'AP: {:.3f}, loss: {:.3f}')
                    print log.format(it, stepsize, p10, ap, loss)
                self._reset_inner_loss()

            qid = np.random.choice(n_queries)
            rel = np.random.choice(self._relevance_matrix[qid].nonzero()[0])

            violator = self._sample_violator(X, Q, qid, rel)

            # Regularizer gradient
            U, V = self._get_regularizer_gradient(Q[qid], X[rel], stepsize)
            (r_a3, r_b1, r_b3, r_a1) = self._gradient_components(U, V)

            if violator:
                U, V = self._get_violator_gradient(Q[qid], X[rel],
                                                   stepsize, violator)
                (v_a3, v_b1, v_b3, v_a1) = self._gradient_components(U, V)
                self.A += np.outer(v_a3, v_b1) + np.outer(r_a3, r_b1)
                self.B += np.outer(v_b3, v_a1) + np.outer(r_b3, r_a1)
                self._update_pseudoinverses(v_a3, v_b1, v_b3, v_a1)
                self._update_pseudoinverses(r_a3, r_b1, r_b3, r_a1)
            else:
                self.A += np.outer(r_a3, r_b1)
                self.B += np.outer(r_b3, r_a1)
                self._update_pseudoinverses(r_a3, r_b1, r_b3, r_a1)

        return self

    def _get_regularizer_gradient(self, query, rel, stepsize):
        return (stepsize * self.lambda_ * query, rel)

    def _get_violator_gradient(self, query, rel, stepsize, violator):
        n_drawn, n_irr, rel_minus_irr = violator
        rank_relevant = n_irr / n_drawn
        loss_value = (self.loss_table[rank_relevant] / self.loss_table[n_irr])
        self.inner_loss.append(loss_value)
        return (stepsize * loss_value * query, rel_minus_irr)

    def _gradient_components(self, U, V):
        a1 = self.A_pinv.dot(U)
        b1 = self.B_pinv.dot(V)
        a2 = self.A.dot(a1)
        s_ = b1.T.dot(a1)
        a3 = a2.dot(-.5 + .375 * s_) + U.dot(1 - .5 * s_)
        b2 = np.dot(V.T.dot(self.B), self.B_pinv)
        b3 = b2.dot(-.5 + .375 * s_) + V.T.dot(1 - .5 * s_)
        return (a3, b1, b3, a1)

    def _update_pseudoinverses(self, a3, b1, b3, a1):
        self.A_pinv = self._rank_one_pinv_update(self.A, self.A_pinv, a3, b1)
        self.B_pinv = self._rank_one_pinv_update(self.B, self.B_pinv, b3, a1)

    def predict(self, Q, X):
        return Q.dot(self.A).dot(X.dot(self.B).T)

    def _validate(self, Q_val, weights, X_val):
        p10 = []
        ap = []
        Y_score = self.predict(Q_val, X_val)
        for x in xrange(self.Y_true.shape[0]):
            p10.append(ranking_precision_score(self.Y_true[x],
                                               Y_score[x]) * weights[x])
            ap.append(average_precision_score(self.Y_true[x],
                                              Y_score[x]) * weights[x])
        return np.sum(p10), np.sum(ap)

    def _initialize_W(self, n, m):
        r"""
        Initialize the parameter matrix in factored form :math:`W = AB^T` as
        ``self.A`` and ``self.B`` and initialize the respective pseudo-inverses
        as ``self.A_pinv`` and ``self.B_pinv``.

        Parameters
        ----------
        k : int
            The rank of W
        n : int
            Dimension for A, such that :math:`A \in \mathbb{R}^{n \times k}`
        m : int
            Dimension for B, such that :math:`B \in \mathbb{R}^{m \times k}`
        """
        # if k > 150:
        #     self.A = np.random.randn(self.n, self.k)
        #     self.B = np.random.randn(self.m, self.k)
        # else:
        self.A = np.vstack((np.eye(self.k), np.zeros((n - self.k, self.k))))
        self.B = np.vstack((np.eye(self.k), np.zeros((m - self.k, self.k))))
        self.A_pinv = sp.linalg.pinv(self.A)
        self.B_pinv = sp.linalg.pinv(self.B)
        self.initialized = True

    def _init_warp_loss_table(self, n_samples):
        if self.loss == 'warp':
            loss_table = np.cumsum(1. / np.arange(1, n_samples))
        elif self.loss == 'auc':
            loss_table = 1. / n_samples * np.ones(n_samples)
        self.loss_table = np.append(0, loss_table)
        # TODO: add p@k loss

    def _rank_one_pinv_update(self, A_, Pinv, c_, d_):
        v_ = Pinv.dot(c_)
        beta_ = 1 + d_.T.dot(v_)
        n_ = Pinv.T.dot(d_)
        w_ = c_ - A_.dot(v_)
        norm_w = w_.T.dot(w_)
        # m_ = d_ - A_.T.dot(Pinv.T.dot(d_))
        # we only deal with full column rank matrices,
        # so norm_m should always be zero
        norm_m = 0
        norm_v = v_.T.dot(v_)
        norm_n = n_.T.dot(n_)

        if abs(beta_) > EPS and norm_m < EPS:
            G_ = np.outer(Pinv.dot(n_) / beta_, w_)
            s_ = beta_ / (norm_w * norm_n + np.square(beta_))
            t_ = norm_w / beta_ * Pinv.dot(n_)
            t_ = s_ * (t_ + v_)
            Gh = np.outer(t_, norm_n * w_ / beta_ + n_)
            G_ = G_ - Gh
        elif abs(beta_) < EPS and norm_w > EPS and norm_m < EPS:
            G_ = -Pinv.dot(n_ / norm_n)
            G_ = G_.dot(n_)
            t_ = np.outer(v_, w_ / norm_w)
            G_ = G_ - t_
        elif abs(beta_) > EPS and norm_w < EPS:
            G_ = -1 / beta_ * np.outer(v_, n_)
        elif abs(beta_) < EPS and norm_w < EPS and norm_m < EPS:
            vh = np.outer(1 / norm_v * v_, v_.T.dot(Pinv))
            nh = 1 / norm_n * np.outer(Pinv.dot(n_), n_)
            s_ = v_.T.dot(Pinv).dot(n_) / (norm_v * norm_n)
            G_ = s_ * np.outer(v_, n_) - vh - nh
        else:
            raise Exception
        return Pinv + G_

    def _reset_inner_loss(self):
        self.inner_loss = []

    def _reset_state(self):
        self.A = None
        self.B = None
        self.A_pinv = None
        self.B_pinv = None
        self.loss_table = None
        self.n_dips = 0
        self.best_A = None
        self.best_B = None
        self.best_ap = 0
        self.initialized = False

    def _sample_violator(self, X, Q, q_id, rel_id):
        """Sample a violator according to the WARP sampling scheme. See [4] for
        details.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        Q : array-like, shape = [n_queries, n_classes]
            Training queries

        q_id : int
            The sampled query id.

        rel_id : int
            The sampled id of a relevant training example for the given
            ``q_id``.

        Returns
        -------
        n_drawn : int
            The number of samples drawn until a violator was found.

        n_irrelevant : int
            The number of irrelevant traning examples for the ``q_id``.

        rel_minus_irr : array, shape = [n_features]
            The vector difference of the relevant and the irrelvant training
            example, :math:`d = x_q^+ - x_q^-`.
        """
        irrelevant_indices = np.where(self._relevance_matrix[q_id] == 0)[0]
        n_irrelevant = len(irrelevant_indices)
        tau = max(1, np.floor(n_irrelevant * self.rank_thresh))

        for n_drawn in xrange(1, int(tau) + 1):
            irr_id = np.random.choice(irrelevant_indices)
            rel_minus_irr = X[rel_id] - X[irr_id]

            if self.predict(Q[q_id], rel_minus_irr) < 1:
                return n_drawn, n_irrelevant, rel_minus_irr

    def _stepsize(self, iteration):
        return self.n0 / (1. + self.n0 * self.n1 * iteration)

    def __repr__(self):
        return ('{}.{}(max_iter={}, k={}, n0={}, n1={}, '
                'rank_thresh={}, lambda_={}, loss=\'{}\', '
                'max_dips={}, valid_interval={})').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.max_iter,
            self.k,
            self.n0,
            self.n1,
            self.rank_thresh,
            self.lambda_,
            self.loss,
            self.max_dips,
            self.valid_interval,
        )


def main():
    from .cross_validation import train_test_split_plus

    X, Y = fetch_cal500()

    (X_train, X_test,
     Y_train, Y_test,
     Q_vec, weights) = train_test_split_plus(X, Y)

    clf = LoretaWARP()
    clf.fit(X_train, Y_train, Q_vec, X_test, Y_test, weights)


if __name__ == '__main__':
    main()
