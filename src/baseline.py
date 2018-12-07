import data_readers as dr
import numpy as np
import pandas as pd

from config import Config
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = Config.SEED

def run_baselines(data, clf):
    X_cols = [col_name for col_name in data.columns if "emb_" in col_name]
    acc = [] 
    auc = []
    for train, dev in dr.get_k_fold_split(data, n_splits=5):
        X_train = train[X_cols]
        X_dev = dev[X_cols]
        y_train = train['is_same']
        y_dev = dev['is_same']
        
        clf.fit(train[X_cols], train['is_same'])
        preds = clf.predict(dev[X_cols])
        probs = [a[1] for a in clf.predict_proba(dev[X_cols])]
        acc.append(accuracy_score(dev['is_same'], preds))
        auc.append(roc_auc_score(dev['is_same'], probs))
    
    return(np.mean(acc), np.mean(auc))        

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
#Dummy:  (0.51063216309525039, 0.51063216309525039)
#GloVe 300d LogReg:  (0.66972953118048151, 0.72767358873453902)
#GloVe 300d LGBM:  (0.70346311835856268, 0.77422512271428801)
#MultiFT .subword only 300d LogReg:  (0.6362570316029813, 0.68654117742617937)
#MultiFT .subword only 300d LGBM:  (0.70661479404430561, 0.77924869419679799)