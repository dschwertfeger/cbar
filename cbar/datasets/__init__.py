"""
The :mod:`cbar.datasets` module includes dataset loading utilities
including methods to load and fetch the CAL500, CAL10k, and Freesound dataset.
"""

# from .base import load_dataset
from .cal10k import fetch_cal10k
from .cal500 import fetch_cal500
from .freesound import load_freesound
from .freesound import load_freesound_queries

__all__ = ['fetch_cal10k',
           'fetch_cal500',
           'load_freesound',
           'load_freesound_queries']
