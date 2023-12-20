#!/bin/bash

# Number of instances to launch
num_instances=3

# Your Python file name
python_file="main.py"

# Python dictionary to pass
python_dictionary="{\"key1\": \"value1\", \"key2\": \"value2\"}"

model_list=(
    "(model_mlp(), \"mlp\", {\"epochs\": 10})"
    "(model_mlp(n_hidden_layers=3, n_units=256), \"mlp\", {\"epochs\": 15})"
    "(model_rnn_simple(), \"rnn\", {\"epochs\": 8})"
)

# Loop to launch instances
for ((i=1; i<=$num_instances; i++)); do
    echo "Launching instance $i..."
    python3 $python_file "{$model_list[$((i - 1))]}" &
done

# Wait for all instances to finish
wait

echo "All instances have completed."
