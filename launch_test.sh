#!/bin/bash

# Your Python file name
python_file="main.py"

python3 $python_file model_mlp "(None,'mlp', {'epochs': 20})" &
python3 $python_file model_mlp "({'n_hidden_layers': 3, 'n_units': 256},'mlp', {'epochs': 20})" &
python3 $python_file model_rnn_simple "(None,'rnn', {'epochs': 20})"& 
python3 $python_file model_cnn_1 "(None,'cnn', {'epochs': 20})" 

# Wait for all instances to finish
wait

echo "All instances have completed."
