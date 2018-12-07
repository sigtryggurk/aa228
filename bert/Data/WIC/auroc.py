# -*- coding: utf-8 -*-
"""
Created on Fri Dec 07 00:41:48 2018

@author: Suvadip Paul
"""

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
labels = pd.read_csv('dev_labels')
scores = pd.read_csv('test_results.tsv', delimiter='\t', index_col=False )
og_labels = labels['label'].apply(lambda x : 0 if x == 'F' else 1 )
preds = scores.iloc[:,0].apply(lambda x : 0 if x <= 0.5 else 1)
accuracy_score(og_labels, preds)
roc_auc_score(og_labels, scores.iloc[:,0])