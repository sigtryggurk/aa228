import data_readers as dr
import numpy as np
import pandas as pd

from config import Config
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = Config.SEED

def run_baselines(data, clf):
    acc = [] 
    for train, dev in dr.get_k_fold_split(data, n_splits=5):
        X_train = train[X_cols]
        X_dev = dev[X_cols]
        y_train = train['is_same']
        y_dev = dev['is_same']
        
        clf.fit(train[X_cols], train['is_same'])
        preds = clf.predict(dev[X_cols])
        acc.append(accuracy_score(dev['is_same'], preds))
    
    return(np.mean(acc))        

data = dr.get_wic()
data = add_embeddings(data)
X_cols = [col_name for col_name in data.columns if "emb_" in col_name]
        
print("Dummy: ", run_baselines(data, DummyClassifier(random_state=SEED))) 
print("LogReg: ", run_baselines(data, LogisticRegression(random_state=SEED)))
print("LGBM: ", run_baselines(data, LGBMClassifier(random_state=SEED)))

#Dummy:  0.510632163095
#LogReg:  0.652402730211
#LGBM:  0.698870114954