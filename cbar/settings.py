from os import path
from datasets.base import create_dir

PROJECT_ROOT = path.dirname(path.abspath(__file__))
RESULTS_DIR = path.join(PROJECT_ROOT, 'diagnostics')

create_dir(RESULTS_DIR)
