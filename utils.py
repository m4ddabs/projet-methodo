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

'''def test_models(model_list, dataset_name, epochs = 5):
    results_path = os.path.join("resultats", dataset_name)
    os.makedirs(results_path, exist_ok=True)
    results = []
    
    for i in range(len(model_list)):
        model, architecture_type = model_list[i]
        
        x_train, y_train, x_test, y_test , original_classes = prepare_data(dataset_name, architecture_type)
        model = build_model(model, input_shape = x_train.shape, n_classes = len(original_classes), architecture_type=architecture_type)
        checkpoint_filepath = os.path.join(results_path, f"model_{architecture_type}_{i+1}.hdf5")
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        print(f"model {i+1}")
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(x=x_train, y=y_train, epochs=20, batch_size=10, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])
        with open(os.path.join(results_path, f"model_{architecture_type}_{i+1}.json"), 'w') as json_file:
            json.dump(history.history, json_file)
        results.append((model, architecture_type))
    return results '''

def test_models(model_list, dataset_name, global_params):
    results_path = os.path.join("resultats", dataset_name)
    os.makedirs(results_path, exist_ok=True)
    results = []

    global_params = {}
    
    for i, (model, architecture_type, model_params) in enumerate(model_list):
        x_train, y_train, x_test, y_test, original_classes = prepare_data(dataset_name, architecture_type)
        model = build_model(model, input_shape=x_train.shape, n_classes=len(original_classes), architecture_type=architecture_type)
        checkpoint_filepath = os.path.join(results_path, f"model_{architecture_type}_{i+1}.hdf5")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

      
        print(f"model {i+1}")

        #Si la clé 'epochs' est présente dans le dictionnaire model_params, alors epochs prend la valeur associée à cette clé (model_params['epochs']
        optimizer = global_params['optimizer'] if 'optimizer' in global_params else 'adam'
        loss = global_params['loss'] if 'loss' in global_params else 'categorical_crossentropy'
        metrics = global_params['metrics'] if 'metrics' in global_params else ['accuracy']

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Récupération du nombre d'époques spécifique au modèle ou utilisation de la valeur globale
        if 'epochs' in model_params:
            epochs = model_params['epochs']
        elif 'epochs' in global_params:
            epochs = global_params['epochs']
        else:
            epochs = 5

        # Ajout d'un EarlyStopping spécifique à chaque modèle si spécifié
        early_stopping_params = model_params.get('early_stopping', None)
        if early_stopping_params:
            early_stopping = EarlyStopping(monitor=early_stopping_params['monitor'] if 'monitor' in early_stopping_params else 'val_loss',
                                           patience=early_stopping_params['patience'] if 'patience' in early_stopping_params else 3,
                                           mode=early_stopping_params['mode'] if 'mode' in early_stopping_params else 'min',
                                           verbose=1)
            callbacks = [model_checkpoint_callback, early_stopping]
        else:
            callbacks = [model_checkpoint_callback]

        # Entraînement du modèle
        history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=global_params['batch_size'] if 'batch_size' in global_params else 10,
                            validation_data=(x_test, y_test), callbacks=callbacks)
        with open(os.path.join(results_path, f"model_{architecture_type}_{i+1}.json"), 'w') as json_file:
            json.dump(history.history, json_file)
        results.append((model, architecture_type))

    
    return results
