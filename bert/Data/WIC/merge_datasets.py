# -*- coding: utf-8 -*-
"""
Created on Thu Dec 06 14:12:30 2018

@author: Suvadip Paul
"""

import pandas as pd
import numpy as np
train = pd.read_csv('train.data.txt', delimiter='\t' )
train_labels = pd.read_csv('train_label.txt')

out = pd.concat([train, train_labels], axis=1)
msk = np.random.rand(len(out)) < 0.8
train_final = out[msk]
dev_final = out[~msk]
train_final.to_csv('train.tsv', sep='\t')
dev_final.to_csv('dev.tsv', sep='\t')