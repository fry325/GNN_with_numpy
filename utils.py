#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:15:24 2021

@author: Gong Dongsheng
"""

import numpy as np
import scipy.sparse as sp


def compute_message_passing_matrix(adj):
    adj = adj + sp.eye(adj.shape[0])
    degree_invsqrt = sp.diags(1.0 / np.sum(adj, axis=1).A1)
    MP = adj @ degree_invsqrt
    return MP


def stratified_sampling(labels, num_per_class=20, val_size=500):
    num_nodes = labels.size
    num_labels = max(labels) + 1
    idx = np.arange(num_nodes)
    train = []
    for curr_label in range(num_labels):
        label_idx = idx[labels == curr_label]
        train.extend(np.random.choice(label_idx, size=num_per_class, replace=False))
    val_test = list(set(list(range(num_nodes))) - set(train))
    val = np.random.choice(val_test, size=val_size, replace=False).tolist()
    test = list(set(val_test) - set(val))
    # print(f"Number of Train, Val, Test: {len(train)} | {len(val)} | {len(test)}")
    return train, val, test


def random_sampling(labels, proportion=[0.6, 0.2, 0.2]):
    num_nodes = labels.size
    ids = list(range(num_nodes))
    np.random.shuffle(ids)
    train_size = int(num_nodes * proportion[0])
    val_size = int(num_nodes * proportion[1])
    train = ids[:train_size]
    val = ids[train_size: train_size + val_size]
    test = ids[train_size + val_size:]
    # print(f"Number of Train, Val, Test: {len(train)} | {len(val)} | {len(test)}")
    return train, val, test