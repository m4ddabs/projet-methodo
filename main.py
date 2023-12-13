from utils import test_models
from model_builder import *
from keras.layers import InputLayer, Dense, SimpleRNN, Input
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential

model_mlp1 = model_mlp()

model_mlp2 = model_mlp(n_hidden_layers=3, n_units=256)

model_rnn1 = model_rnn_simple()



#model_list = [(model_mlp1, "mlp"), (model_mlp2, "mlp"), (model_rnn1, "rnn")]

#results = test_models(model_list=model_list,dataset_name="TwoPatterns")
## Mettre none dans la troisième valeur du tuple au lieu du dico si aucun paramètre va 
## etre préciser pour le fit. 
model_list = [
    (model_mlp1, "mlp", {'epochs': 10}),
    (model_mlp2, "mlp", {'epochs': 15}),
    (model_rnn1, "rnn", {'epochs': 8}),
]


results = test_models(model_list=model_list, dataset_name="TwoPatterns")


#### Plots
results_hist = [tup[0].history.history for tup in results]
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0, 0].plot(results_hist[0]["val_accuracy"])
axs[0, 0].set_title('Modele sequentiel simple')
axs[0, 1].plot(results_hist[1]["val_accuracy"])
axs[0, 1].set_title('Modele sequentiel avec 3 hidden layers')
axs[1, 0].plot(results_hist[2]["val_accuracy"])
axs[1, 0].set_title('Modele RNN simple')


for ax in axs.flat:
    ax.set(xlabel='Epochs', ylabel="Validation Accuracy")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()