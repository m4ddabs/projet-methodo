from utils import test_models
from model_builder import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

# model_mlp1 = model_mlp()

# model_mlp2 = model_mlp(n_hidden_layers=3, n_units=256)

# model_rnn1 = model_rnn_simple()


# model_list = [
#     (model_mlp1, "mlp", {'epochs': 10}),
#     (model_mlp2, "mlp", {'epochs': 15}),
#     (model_rnn1, "rnn", {'epochs': 8}),
# ]



# Mettre none dans la troisième valeur du tuple au lieu du dico si aucun paramètre va 
# etre préciser pour le fit. 


# Check if a list argument is provided
if len(sys.argv) > 2:
    # Extract the function name and arguments
    function_name = sys.argv[1]
    args = sys.argv[2]

    # Check if the function exists
    if function_name in globals() and callable(globals()[function_name]):
        # Call the specified function with the provided arguments
        params_model,model_type,params_fit = ast.literal_eval(args)
        if params_model is not None:
            model = (globals()[function_name](**params_model), model_type, params_fit)
        else:
            model = (globals()[function_name](), model_type, params_fit)
    else:
        print(f"Function '{function_name}' not found or not callable.")
else:
    print("Insufficient arguments.")

model_list =[model]

results = test_models(model_list=model_list, dataset_name="Libras")