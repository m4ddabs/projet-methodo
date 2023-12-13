import tslearn # pour les séries
from tslearn.datasets import UCR_UEA_datasets
import keras_core as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import os
import json
from model_builder import *
import random

def prepare_data(dataset_name, architecture_type):

    data_loader = UCR_UEA_datasets()
    x_train, y_train, x_test, y_test = data_loader.load_dataset(dataset_name)
    # Pour les deux architecture RNN et CNN , on doit avoir une dataset de type (n,T,c)
    # Si l'architecture est RNN
    if architecture_type == 'rnn':
        # x_train = x_train.reshape((x_train.shape[0], -1, 1))
        # x_test = x_test.reshape((x_test.shape[0], -1, 1))
        print("rnn")

    # Si l'architecture est CNN
    elif architecture_type == 'cnn':
        print("cnn")
        
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
     
   
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test= label_encoder.transform(y_test)
    
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    return x_train, y_train, x_test, y_test , label_encoder.classes_



def creation_app_val(x_train, y_train, ratio=0.15):
    
    data_train = list(zip(x_train, y_train))
    random.shuffle(data_train)
    num_val = int(ratio * len(data_train))
    num_app = len(data_train) - num_val

   
    data_app = data_train[:num_app]
    data_val = data_train[num_app:num_app + num_val]
    
    
    x_app, y_app = zip(*data_app)
    x_val, y_val = zip(*data_val)
    x_app = np.asarray(x_app)
    y_app = np.asarray(y_app)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    
    
    return x_app, y_app, x_val, y_val

def test_models(model_list, dataset_name):
    results_path = os.path.join("resultats", dataset_name)
    os.makedirs(results_path, exist_ok=True)
    results = []
    for i, (model, architecture_type, model_params) in enumerate(model_list):

        x_train, y_train, x_test, y_test, original_classes = prepare_data(dataset_name, architecture_type)
        x_app, y_app, x_val, y_val=creation_app_val(x_train, y_train)

        model = build_model(model, input_shape=x_train.shape, n_classes=len(original_classes), architecture_type=architecture_type)
        checkpoint_filepath = os.path.join(results_path, f"model_{architecture_type}_{i+1}.hdf5")

        ## Callbacks
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience= 3, mode='min', verbose=1)

        callbacks = [model_checkpoint_callback, early_stopping]
      
        print(f"model {i+1}")

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Entraînement du modèle
        if model_params == None:
            history = model.fit(x=x_app, y=y_app, epochs=10, batch_size=32, validation_data=(x_val, y_val), 
                        callbacks=callbacks)
        else: 
            history = model.fit(x=x_app, y=y_app, validation_data=(x_val, y_val), 
                        callbacks=callbacks, **model_params)
        with open(os.path.join(results_path, f"model_{architecture_type}_{i+1}.json"), 'w') as json_file:
            json.dump(history.history, json_file)
        results.append((model, architecture_type))

    
    return results
