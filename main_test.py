from utils import test_models
from model_builder import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

with open("datasets.txt", 'r', encoding='utf-8') as fp:
    datasets = []
    for line  in fp.readlines():
        datasets.append(line.strip('\n'))

model_mlp1 = model_mlp()

model_mlp2 = model_mlp(n_hidden_layers=3, n_units=256)

model_rnn1 = model_rnn_simple()

model_cnn1 = model_cnn_1()

# model_list = [
#     (model_mlp1, "mlp", {'epochs': 10}),
#     (model_mlp2, "mlp", {'epochs': 15}),
#     (model_rnn1, "rnn", {'epochs': 8}),
# ]



for dataset in datasets:
    model_cnn1 = model_cnn_1()
    model_list =[(model_cnn1, "cnn", {'epochs': 10})]
    test_models(model_list=model_list, dataset_name=dataset)

# test_models(model_list=model_list, dataset_name='Beef')