import data_readers as dr
import numpy as np
import pandas as pd

from config import Config
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = Config.SEED

def run_baselines(data, clf):
    X_cols = [col_name for col_name in data.columns if "emb_" in col_name]
    acc = []
    auc = []
    train, dev = train_test_split(data, test_size=0.2, random_state=SEED)

    X_train = train[X_cols]
    X_dev = dev[X_cols]
    y_train = train['is_same']
    y_dev = dev['is_same']

    clf.fit(train[X_cols], train['is_same'])
    preds = clf.predict(dev[X_cols])
    probs = [a[1] for a in clf.predict_proba(dev[X_cols])]
    acc = accuracy_score(dev['is_same'], preds)
    auc = roc_auc_score(dev['is_same'], probs)

    return(acc, auc)

data = dr.get_wic()
data = dr.add_embeddings(data)
print("GloVe 300d")
print("Dummy: ", run_baselines(data, DummyClassifier(random_state=SEED)))
print("LogReg: ", run_baselines(data, LogisticRegression(random_state=SEED)))
print("LGBM: ", run_baselines(data, LGBMClassifier(random_state=SEED)))

print("Multisense FastText (using only .subword)")
data_multift = dr.get_wic()
data_multift = dr.add_embeddings(
        data_multift, embed_file=Config.MULTIFT_EMBED_FILE, cased=True)
print("Dummy: ", run_baselines(data_multift, DummyClassifier(random_state=SEED)))
print("LogReg: ", run_baselines(data_multift, LogisticRegression(random_state=SEED)))
print("LGBM: ", run_baselines(data_multift, LGBMClassifier(random_state=SEED)))

#Model (Accuracy, AUC)
#GloVe 300d
#Dummy:  (0.49343832020997375, 0.49337262200165433)
#LogReg:  (0.67716535433070868, 0.73230631375792665)
#LGBM:  (0.70341207349081369, 0.76380962227736426)
#Multisense FastText (using only .subword)
#Done. 2677466  words loaded!
#Dummy:  (0.49343832020997375, 0.49337262200165433)
#LogReg:  (0.63188976377952755, 0.68891990625861588)
#LGBM:  (0.71850393700787396, 0.77560311552247041)
