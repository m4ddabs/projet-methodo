#!/bin/bash

# Your Python file name
python_file="main.py"

python3 $python_file model_mlp "(None,'mlp', {'epochs': 100})" model_mlp_simple &
python3 $python_file model_mlp "({'n_hidden_layers': 3, 'n_units': 256},'mlp', {'epochs': 100})" model_mlp_3L &
python3 $python_file model_rnn_simple "(None,'rnn', {'epochs': 100})" model_rnn_simple & 
python3 $python_file model_cnn_1 "(None,'cnn', {'epochs': 100})" model_cnn_simple
# python3 $python_file model_lstm "(None,'rnn', {'epochs': 20})" model_2lstm_2dense

# Wait for all instances to finish
wait

echo "All instances have completed."
