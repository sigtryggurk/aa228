from os.path import dirname, realpath, join

class Config:
    BASE_DIR = realpath(join(dirname(realpath(__file__)), '..'))
    RUNS_DIR = join(BASE_DIR, "runs")
    DATA_DIR = join(BASE_DIR, "data")
    WIC_SAMPLES_FILE = join(DATA_DIR, "wic.train.data.txt")
    WIC_DEV_SAMPLES_FILE = join(DATA_DIR, "wic.dev.data.txt")
    WIC_LABELS_FILE = join(DATA_DIR, "wic.train.gold.txt")
    EMBED_DIR = join(BASE_DIR, "embed")
    GLOVE_FILE = join(EMBED_DIR, "glove.6B.300d.txt")
    GLOVE_WIKICLEAN_FILE = join(EMBED_DIR, "glove.wikiclean.50d.txt")
    MULTIFT_BASE = join(EMBED_DIR, "mv-wacky_e10_d300_vs2e-4_lr1e-5_mar1")
    MULTIFT_WORDS_FILE = join(
            EMBED_DIR, "mv-wacky_e10_d300_vs2e-4_lr1e-5_mar1.words")
    MULTIFT_EMBED_FILE = join(
            EMBED_DIR, "mv-wacky_e10_d300_vs2e-4_lr1e-5_mar1.subword.npy")
    VOCAB_FILE = join(EMBED_DIR, "vocab")
    DUAL_SENSE_FILE = join(EMBED_DIR, "dual.vec")
    
    SEED = 42
