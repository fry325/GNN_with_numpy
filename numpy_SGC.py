#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:22:40 2021

@author: Gong Dongsheng
"""

import numpy as np
from data.io import load_dataset
from utils import stratified_sampling, random_sampling
from sklearn.preprocessing import OneHotEncoder

graph_name = "pubmed"
stratified = True
num_layer = 2
num_epochs = 10000
patience = 100
lr = 0.2
seed = 10000
np.random.seed(seed)


def softmax(x):
    dimx = x.shape[1]
    x = x - x.max(axis=1, keepdims=True)
    expx = np.exp(x)
    return expx / np.matmul(expx, np.ones((dimx, dimx)))


def xavier_uniform(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(-bound, bound, shape)


class NumpySGC:
    def __init__(self, n_features, n_labels):
        self.W = xavier_uniform((n_features, n_labels))
    
    def forward_propagate(self, H):
        return softmax(np.matmul(H, self.W))
    
    def backward_propagate(self, H, y, y_pred, mask, optimizer):
        gradient =  - H.T @ (y - (mask @ y_pred))
        self.W = optimizer.update(gradient, self.W)
        
        
class Adam:
    def __init__(self, lr, w_shape):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.v = np.zeros(w_shape)
        self.s = np.zeros(w_shape)
        self.t = 0
        self.eps = 1e-08
        
    def update(self, gradient, w):
        self.t += 1
        self.v = self.beta1 * self.v + (1 - self.beta1) * gradient
        self.s = self.beta2 * self.s + (1 - self.beta2) * np.power(gradient, 2)
        v = self.v / (1 - self.beta1 ** self.t)
        s = self.s / (1 - self.beta2 ** self.t)
        g_prime = (self.lr * v) / (np.sqrt(s) + self.eps)
        return w - g_prime


def run(graph):
    num_nodes = graph.labels.size
    num_labels = max(graph.labels) + 1
    num_features = graph.attr_matrix.shape[1]
    stratified_func = stratified_sampling if stratified else random_sampling
    train, val, test = stratified_func(graph.labels)
    
    X = graph.attr_matrix
    A = graph.adj_matrix
    H = X
    for layer in range(num_layer):
        H = A @ H
    H = np.array(H.todense())
    
    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(graph.labels.reshape(-1, 1))
    mask = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if i in train:
            mask[i, i] = 1
    y = mask @ y
    
    model = NumpySGC(num_features, num_labels)
    optimizer = Adam(lr=lr, w_shape=model.W.shape)
    best_val_acc, best_test_acc, best_epoch = 0.0, 0.0, 0
    y_pred = model.forward_propagate(H)
    for epoch in range(num_epochs):
        model.backward_propagate(H, y, y_pred, mask, optimizer)
        
        y_pred = model.forward_propagate(H)
        val_acc = sum(graph.labels[val] == y_pred[val].argmax(axis=1)) / len(val)
        test_acc = sum(graph.labels[test] == y_pred[test].argmax(axis=1)) / len(test)
        # print("Epoch {} | Val ACC {} %".format(epoch + 1, np.round(val_acc * 100, 4)))
        if val_acc > best_val_acc:
            best_val_acc, best_test_acc, best_epoch = val_acc, test_acc, epoch
        if epoch >= patience + best_epoch:
            break
    
    print("Test ACC: {} %".format(np.round(best_test_acc * 100, 4)))
    return best_test_acc
    
    
if __name__ == "__main__":
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True)
    accs = []
    for i in range(10):
        acc = run(graph)
        accs.append(acc)
    print("Accuracy: {} ± {} %".format(
            np.round(np.mean(accs) * 100, 2),
            np.round(np.std(accs) * 100, 2)
        )
    )