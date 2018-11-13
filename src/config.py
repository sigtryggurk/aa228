from os.path import dirname, realpath, join
from pathlib import Path

class Config:
    BASE_DIR = Path(realpath(join(dirname(realpath(__file__)), '..')))
    RUNS_DIR = Path(join(BASE_DIR, "runs"))
    DATA_DIR = Path(join(BASE_DIR, "data"))
    WIC_SAMPLES_FILE = Path(join(DATA_DIR, "wic.train.data.txt"))
    WIC_LABELS_FILE = Path(join(DATA_DIR, "wic.train.gold.txt"))
