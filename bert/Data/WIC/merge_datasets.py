# -*- coding: utf-8 -*-
"""
Created on Thu Dec 06 14:12:30 2018

@author: Suvadip Paul
"""

import pandas as pd
import numpy as np
import sklearn.model_selection 

train = pd.read_csv('train.data.txt', delimiter='\t' )
train_labels = pd.read_csv('train_label.txt')

out = pd.concat([train, train_labels], axis=1)
train_final, dev_final = sklearn.model_selection.train_test_split(out, test_size=0.2, random_state=42)
train_final.to_csv('train.tsv', sep='\t')
dev_final.to_csv('dev.tsv', sep='\t')

train_label, dev_label = sklearn.model_selection.train_test_split(train_labels, test_size=0.2, random_state=42)
dev_label.to_csv('dev_labels')
test = pd.read_csv('test.tsv', delimiter='\t')
#test.to_csv('test.tsv', sep='\t')

