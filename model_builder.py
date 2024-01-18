import keras_core as keras
from keras.models import Sequential
from keras.layers import InputLayer, Input, Dense, Conv1D, MaxPool1D, Flatten, Dropout, SimpleRNN, Bidirectional, LSTM



##Architecture inspirée de https://www.omdena.com/blog/time-series-classification-model-tutorial
## Nous avons laissé de coté la partie de réseaux qui traite les variables catégoriques. 
def model_lstm_bi():
    layers = [
        Bidirectional(LSTM(units=20, activation="relu", return_sequences=True)),
        Flatten(),
        Dense(units=100, activation="relu")
    ]
    return layers

def model_cnn_1(padding = "valid", strides = None, pool_size = 2, kernel_size = 5):
    layers = [
        Conv1D(filters=6, kernel_size=kernel_size, activation='relu', padding=padding),
        MaxPool1D(pool_size=pool_size, padding='valid'),
        Conv1D(filters=16, kernel_size=kernel_size, activation='relu', padding=padding),
        MaxPool1D(pool_size=pool_size, padding=padding),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=120, activation='relu'),
        Dense(units=84, activation='relu')
    ]
    return layers

def model_rnn_simple(n_units=64):
    layers = [SimpleRNN(n_units, return_sequences=False)]
    return layers

## Fonction model_mlp: 
## Definis un modele mlp pour la classification de series temporelles. 
def model_mlp(n_hidden_layers=0, n_units=64, activation ="relu"):
    layers = []
    if n_hidden_layers > 0:
        for i in range(n_hidden_layers):
            layers.append(Dense(units=n_units, activation=activation))
    return layers


## This function only builds sequential models 
def build_model(layers, input_shape, n_classes, architecture_type):
    if architecture_type == 'mlp':
        shape = (input_shape[1], ) # Nous laissons de cote le batch size
    if architecture_type in ['rnn', 'cnn']:
        shape = (input_shape[1], input_shape[2]) # Nous laissons de cote le batch size
    layers.insert(0,InputLayer(input_shape=shape))
    layers.append(Dense(units=n_classes, activation="softmax"))
    return Sequential(layers)



if __name__ == "__main__":
    print("Hello World")