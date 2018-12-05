import multift
import numpy as np
import pandas as pd

from config import Config
from sklearn.model_selection import StratifiedKFold

def get_wic(samples_f=Config.WIC_SAMPLES_FILE, labels_f=Config.WIC_LABELS_FILE):
    samples = pd.read_csv(samples_f, delimiter='\t', header=None, names=["w", "pos", "indices", "ctx1", "ctx2"])
    indices = samples.indices.apply(lambda s: tuple(map(int, s.split('-'))))
    samples.drop(["indices"], inplace=True, axis=1)

    start1, start2 = list(zip(*indices))
    samples = samples.assign(start1=start1)
    samples = samples.assign(start2=start2)

    if labels_f is not None:
      labels = pd.read_csv(labels_f, header=None, squeeze=True, converters={0 :lambda t: True if t=="T" else False})
      data = samples.assign(is_same = labels)
    else:
      data = samples
    return data

def get_embedding_model(embed_file=Config.GLOVE_FILE, dim=300):
    if embed_file != Config.DUAL_SENSE_FILE:
        fails = 0
        #with open(embed_file, encoding="utf8" ) as f:
        with open(embed_file ) as f:
           content = f.readlines()
        model = {}
        for line in content:
            splitLine = line.split()
            if len(splitLine) == dim+1:
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            else:
                fails += 1
        print ("Done.", len(model), " words loaded!")
        print("Skipped " + str(fails) + " words!")
    else:
        with open(Config.VOCAB_FILE) as f:
            words = f.readlines()

        vecs = np.load(Config.DUAL_SENSE_FILE + ".npy")

        model = {word[:-1] : vecs[i].flatten() for i, word in enumerate(words)}
        model['unk'] = model['UNK'] = np.zeros_like(vecs[0].flatten())
    return model


def add_embeddings(data, embed_file=Config.GLOVE_FILE, cased=False):
    print("Getting embedding model from %s" % embed_file)
    model = get_embedding_model(embed_file)
    vect_1 = []
    vect_2 = []
    if 'unk' not in model:
        if 'UNK' in model:
            model['unk'] = model['UNK']
        
    word_c = lambda x, cased: x if cased else x.lower()
   
    print("Computing ctx1 vectors")
    for sent1 in data['ctx1']:
        vect_1.append(np.mean([model.get(word_c(word, cased), model['unk']) for word in sent1.split()], axis=0))
    print("Computing ctx2 vectors")
    for sent2 in data['ctx2']:
        vect_2.append(np.mean([model.get(word_c(word, cased), model['unk']) for word in sent2.split()], axis=0))
    print("Computing target word vectors")
    vect_w = [model.get(word_c(word, cased), model['unk']) for word in data['w']]
    vect = np.hstack((vect_1, vect_2, vect_w))
    columns = ["emb_" + str(i) for i in range(len(vect[0]))]
    data = data.join(pd.DataFrame(vect, columns=columns))
    return data

def get_k_fold_split(data, n_splits=2):
    assert n_splits >= 2

    skf = StratifiedKFold(n_splits=n_splits, random_state=Config.SEED, shuffle=False)

    for train_index, test_index in skf.split(np.zeros(len(data)), data.is_same.values):
        yield data.iloc[train_index], data.iloc[test_index]

if __name__ == "__main__":
    data = get_wic()
    data = add_embeddings(data)
    for train, test in get_k_fold_split(data, n_splits=5):
        print(len(train), len(test))
