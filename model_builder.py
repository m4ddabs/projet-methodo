import keras
from keras import layers
from keras.models import Sequential
from keras.layers import InputLayer, Input, Dense, Conv1D, MaxPool1D, Flatten, Dropout, SimpleRNN, Bidirectional, LSTM, GRU, BatchNormalization
from keras import backend as K


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

class attention_custom(keras.layers.Layer):
    def __init__(self):
        # Nothing special to be done here
        super(attention_custom, self).__init__()

    def build(self, input_shape):
        # Define the shape of the weights and bias in this layer
        # As we discussed the layer has just 1 lonely neuron
        # We discussed the shapes of the weights and bias earlier
        self.inp_dim = input_shape[-1]
        self.seq_length = input_shape[-2]  #timesteps per time series
        num_units = 1
        self.w = self.add_weight(shape=(self.inp_dim,num_units),initializer='normal', name="weight_1")
        self.b = self.add_weight(shape=(self.seq_length,num_units),initializer='zero', name='bias_1')
        super(attention_custom, self).build(input_shape)

    def call(self, x):
        # x is the input tensor of inp_dim dimensions        # Below is the main processing done during training
        # K is the Keras Backend import
        e = K.tanh(K.dot(x,self.w)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        # return the outputs. 'a' is the set of seq_length attention weights
        # the second variable is the 'attention adjusted o/p state'
        return a, K.sum(output, axis=1)




def model_lstm_bi_attention():
    layers = [
        Bidirectional(LSTM(units=128, activation="relu", return_sequences=True, name="lstm1")),
        attention_custom(),
        Dense(units=100, activation="relu", name="dense1")
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

def model_cnn2_dropout(padding = "valid", strides = None, pool_size = 2, kernel_size = 5, dropout_val=0.3):
    layers = [
        Conv1D(filters=6, kernel_size=kernel_size, activation='relu', padding=padding),
        MaxPool1D(pool_size=pool_size, padding='valid'),
        Conv1D(filters=16, kernel_size=kernel_size, activation='relu', padding=padding),
        MaxPool1D(pool_size=pool_size, padding=padding),
        Flatten(),
        Dropout(dropout_val),
        Dense(units=256, activation='relu'),
        Dropout(dropout_val),
        Dense(units=120, activation='relu'),
        Dropout(dropout_val),
        Dense(units=84, activation='relu'),
        Dropout(dropout_val)
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
def model_mlp(n_hidden_layers=0, n_units=64, activation ="relu", dropout = False, dropout_val = 0.3):
    layers = []
    if n_hidden_layers > 0:
        for i in range(n_hidden_layers):
            layers.append(Dense(units=n_units, activation=activation))
            if dropout == True:
                layers.append(Dropout(dropout_val))
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

## This function builds models 
def build_model(model, input_shape, n_classes, architecture_type):
    if architecture_type == 'mlp':
        shape = (input_shape[1], ) # Nous laissons de cote le batch size
    if architecture_type in ['rnn', 'cnn', 'rnn-att']:
        shape = (input_shape[1], input_shape[2]) # Nous laissons de cote le batch size
        if architecture_type == 'rnn-att':
          input = Input(shape=shape)
          x = model[0](input)
          for i in range(1,len(model)):
            if i == 1:
              a, x = model[i](x)        ## a = poids d'attention, regarder la function call de la class attention custom pour comprendre
            else:
              x = model[i](x)
          output = Dense(units=n_classes, activation="softmax")(x)
          return keras.Model(inputs=input, outputs=output)          

    if architecture_type == "transformer":
        shape = (input_shape[1], input_shape[2])
        head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout , mlp_dropout = model
        return build_model_transformer(shape, n_classes, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)
    model.insert(0,InputLayer(input_shape=shape))
    model.append(Dense(units=n_classes, activation="softmax"))
    return Sequential(layers)




if __name__ == "__main__":
    print("Hello World")