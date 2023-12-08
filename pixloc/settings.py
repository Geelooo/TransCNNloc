from importlib.resources import path
from pathlib import Path

root1 = Path(__file__).parent.parent  # top-level directory
TRAINING_PATH = root1 / 'outputs/training/'  # training checkpoints

root = Path(__file__).parent.parent.parent.parent
DATA_PATH = root / 'datasets/'  # datasets and pretrained weights
LOC_PATH = DATA_PATH   # localization logs
EVAL_PATH = DATA_PATH   # evaluation results

