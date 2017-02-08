CBAR: Content-Based Audio Retrieval in Python
=============================================

CBAR is a Python package for content-based audio retrieval with text queries.

It contains two retrieval methods. The Passive-Aggressive Model for Image Retrieval (PAMIR) was initially
developed in the context of an image retrieval application [1]_ but has been
proven to work equally well for audio retrieval applications [2]_.

The second approach combines on a Low-Rank Retraction Algorithm (LORETA) [3]_
and the Weighted Approximate-Rank Pairwise loss (WARP loss) [4]_ to efficiently
infer the model parameters. A similar algorithm, constrained to the context
of finding similar items of the same kind (similary search), has been shown to
work well on image and audio datasets [5]_.


Getting started
---------------

Jump straight to the :doc:`CAL500 quickstart <notebooks/quickstart>` guide
if you are impatient.


Installation
------------

The latest release of CBAR can be installed from PyPI using ``pip``.

.. code:: bash

    pip install cbar


Dependencies
------------

CBAR is tested on Python 2.7 and depends on NumPy, SciPy, Pandas, NLTK, and
scikit-learn. See ``setup.py`` for version information.


Documentation
-------------

https://dschwertfeger.github.io/cbar


On GitHub
---------

https://github.com/dschwertfeger/cbar


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

.. [3] Shalit, U., Weinshall, D. and Chechik, G., 2012. `Online learning in
        the embedded manifold of low-rank matrices.
        <http://www.jmlr.org/papers/volume13/shalit12a/shalit12a.pdf>`_
        Journal of Machine Learning Research, 13(Feb), pp.429-458.

.. [4] Weston, J., Bengio, S. and Usunier, N., 2010. `Large scale image
        annotation: learning to rank with joint word-image embeddings.
        <https://research.google.com/pubs/archive/35780.pdf>`_
        Machine learning, 81(1), pp.21-35.

.. [5] Lim, D. and Lanckriet, G., 2014. `Efficient Learning of Mahalanobis
        Metrics for Ranking.
        <http://www.jmlr.org/proceedings/papers/v32/lim14.pdf>`_
        In Proceedings of The 31st International Conference on Machine Learning
        (pp. 1980-1988).
