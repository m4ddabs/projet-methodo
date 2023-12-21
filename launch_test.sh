#!/bin/bash

# Your Python file name
python_file="main.py"

python3 $python_file model_mlp "(None,'mlp', {'epochs': 10})" &
python3 $python_file model_mlp "({'n_hidden_layers': 3, 'n_units': 256},'mlp', {'epochs': 15})" &
python3 $python_file model_rnn_simple "(None,'rnn', {'epochs': 8})"

# Wait for all instances to finish
wait

echo "All instances have completed."
