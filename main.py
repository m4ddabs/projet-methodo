from utils import prepare_data, test_models
from keras.layers import InputLayer, Dense, SimpleRNN, Input
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt

model_mlp1 = Sequential([
    InputLayer(input_shape=(90,)),
    Dense(units=15, activation="softmax") # softmax : garantit des propab sont entre 0 et1 , et la somme des probas egal 1
])


model_mlp2 = Sequential([
    InputLayer(input_shape=(90,)),
    Dense(units=256, activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=256, activation="sigmoid"),
    Dense(units=15, activation="softmax")
])

def make_model_rnn(tslength):
    input_layer = Input(shape=(tslength,2))
    srnn1 = SimpleRNN(64, return_sequences=False)(input_layer)
    output_layer = Dense(units=15, activation="softmax")(srnn1)
    model = Model(input_layer, output_layer)
    return model

model_rnn = make_model_rnn(45)

model_list = [(model_mlp1, "mlp"), (model_mlp2, "mlp"), (model_rnn, "rnn")]

results = test_models(model_list=model_list,dataset_name="Libras")


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