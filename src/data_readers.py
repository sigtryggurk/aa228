import numpy as np
import pandas as pd

from config import Config
from sklearn.model_selection import StratifiedKFold

def get_wic():
    samples = pd.read_csv(Config.WIC_SAMPLES_FILE, delimiter='\t', header=None, names=["w", "pos", "indices", "ctx1", "ctx2"])
    indices = samples.indices.apply(lambda s: tuple(map(int, s.split('-'))))
    samples.drop(columns="indices", inplace=True)

    start1, start2 = list(zip(*indices))
    samples = samples.assign(start1=start1)
    samples = samples.assign(start2=start2)

    labels = pd.read_csv(Config.WIC_LABELS_FILE, header=None, squeeze=True, converters={0 :lambda t: True if t=="T" else False})
    data = samples.assign(is_same = labels)
    return data

def get_k_fold_split(data, n_splits=2):
    assert n_splits >= 2

    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=False)

    for train_index, test_index in skf.split(np.zeros(len(data)), data.is_same.values):
        yield data.iloc[train_index], data.iloc[test_index]

if __name__ == "__main__":
    data = get_wic()
    for train, test in get_k_fold_split(data, n_splits=3):
        print(len(train), len(test))

