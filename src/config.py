from os.path import dirname, realpath, join
from pathlib import Path

class Config:
    BASE_DIR = Path(realpath(join(dirname(realpath(__file__)), '..')))
    RUNS_DIR = Path(join(BASE_DIR, "runs"))
    DATA_DIR = Path(join(BASE_DIR, "data"))
    WIC_SAMPLES_FILE = Path(join(DATA_DIR, "wic.train.data.txt"))
    WIC_LABELS_FILE = Path(join(DATA_DIR, "wic.train.gold.txt"))
    EMBED_DIR = Path(join(BASE_DIR, "embed"))
    GLOVE_FILE = Path(join(EMBED_DIR, "glove.6B.300d.txt"))
    GLOVE_WIKICLEAN_FILE = Path(join(EMBED_DIR, "glove.wikiclean.50d.txt"))
    MULTIFT_WORDS_FILE = Path(join(
            EMBED_DIR, "mv-wacky_e10_d300_vs2e-4_lr1e-5_mar1.words"))
    MULTIFT_EMBED_FILE = Path(join(
            EMBED_DIR, "mv-wacky_e10_d300_vs2e-4_lr1e-5_mar1.subword.npy"))
    
    SEED = 42