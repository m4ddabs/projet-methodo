import tslearn # pour les séries
from tslearn.datasets import UCR_UEA_datasets
import keras_core as keras
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

def prepare_data(dataset_name, architecture_type):
    data_loader = UCR_UEA_datasets()
    x_train, y_train, x_test, y_test = data_loader.load_dataset(dataset_name)
    # Pour les deux architecture RNN et CNN , on doit avoir une dataset de type (n,T,c)
    # Si l'architecture est RNN
    if architecture_type == 'rnn':
        x_train = x_train.reshape((x_train.shape[0], -1, 1))
        x_test = x_test.reshape((x_test.shape[0], -1, 1))

    # Si l'architecture est CNN
    elif architecture_type == 'cnn':
        x_train = x_train.reshape((x_train.shape[0], -1, 1, 1))
        x_test = x_test.reshape((x_test.shape[0], -1, 1, 1))
    # Pour l'architecture MLP, notre dataset doit etre sous la forme (n,T*c) avec un reshape de T c 
    # Si l'architecture est MLP
    elif architecture_type == 'mlp':
        # Aplatir les données pour les réseaux de neurones à couches denses (MLP)
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

    scaler = MinMaxScaler()
    scaler.fit(x_train.reshape(-1, 1))  
    x_train = scaler.transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)

    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train - 1, num_classes=num_classes)
    y_test = to_categorical(y_test - 1, num_classes=num_classes)
    return x_train, y_train, x_test, y_test
