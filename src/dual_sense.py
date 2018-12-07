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

def run_model(data, clf):
    X_cols = [col_name for col_name in data.columns if "emb_" in col_name]
    acc = []
    auc = []
    train, dev = train_test_split(data, test_size=0.2, random_state=SEED)

    X_train = train[X_cols]
    X_dev = dev[X_cols]
    y_train = train.is_same
    y_dev = dev.is_same

    clf.fit(train[X_cols], train.is_same)
    preds = clf.predict(dev[X_cols])
    probs = [a[1] for a in clf.predict_proba(dev[X_cols])]
    acc = accuracy_score(dev.is_same, preds)
    auc = roc_auc_score(dev.is_same, probs)

    return acc, auc

if __name__ == "__main__":
    print("Multisense FastText (using only dual sense)")
    data_multift = dr.get_wic()
    data_multift = dr.add_embeddings(data_multift, embed_file=Config.MULTIFT_BASE, cased=True)
    print("Dummy:\tacc: {}\n\troc_auc: {}".format(*run_model(data_multift, DummyClassifier(random_state=SEED))))
    print("LogReg:\tacc: {}\n\troc_auc: {}".format(*run_model(data_multift, LogisticRegression(random_state=SEED))))
    print("LGBM:\tacc: {}\n\troc_auc: {}".format(*run_model(data_multift, LGBMClassifier(random_state=SEED))))
