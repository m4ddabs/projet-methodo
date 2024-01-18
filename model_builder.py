import keras
from keras import layers
from keras.models import Sequential
from keras.layers import InputLayer, Input, Dense, Conv1D, MaxPool1D, Flatten, Dropout, SimpleRNN, Bidirectional, LSTM, GRU, BatchNormalization


##https://keras.io/examples/timeseries/timeseries_classification_transformer/
def model_transformer(head_size=256,num_heads=4,ff_dim=4,num_transformer_blocks=4,mlp_units=[128],dropout=0,mlp_dropout=0):
    return (head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)

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

def model_rnn_gru_avec_bn(n_units=64):
    layers = [
        Bidirectional(GRU(n_units, return_sequences=False)),
        BatchNormalization(),
        Dense(units=256, activation='relu'),
        BatchNormalization(),
        Dense(units=120, activation='relu'),
        BatchNormalization(),
        Dense(units=84, activation='relu')
    ]
    return layers



## Fonction model_mlp: 
## Definis un modele mlp pour la classification de series temporelles. 
def model_mlp(n_hidden_layers=0, n_units=64, activation ="relu"):
    layers = []
    if n_hidden_layers > 0:
        for i in range(n_hidden_layers):
            layers.append(Dense(units=n_units, activation=activation))
    return layers

def model_mlp_4l():
        layers = [
        Dense(500, activation='relu'),
        Dropout(0.1),
        Dense(500, activation='relu'),
        Dropout(0.2),
        Dense(500, activation='relu'),
        Dropout(0.2),
        Dense(500, activation='relu'),
        Dropout(0.3)
        ]
        
        
        return layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model_transformer(input_shape,n_classes,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout=0,mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

## This function only builds sequential models 
def build_model(model, input_shape, n_classes, architecture_type):
    if architecture_type == 'mlp':
        shape = (input_shape[1], ) # Nous laissons de cote le batch size
    if architecture_type in ['rnn', 'cnn']:
        shape = (input_shape[1], input_shape[2]) # Nous laissons de cote le batch size
    if architecture_type == "transformer":
        shape = (input_shape[1], input_shape[2])
        head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout , mlp_dropout = model
        return build_model_transformer(shape, n_classes, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)
    model.insert(0,InputLayer(input_shape=shape))
    model.append(Dense(units=n_classes, activation="softmax"))
    return Sequential(layers)



if __name__ == "__main__":
    print("Hello World")