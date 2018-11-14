import pandas as pd

from config import Config

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

if __name__ == "__main__":
    data = get_wic()
    print(len(data))
    print(sum(data.is_same))

