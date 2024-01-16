from utils import test_models
from model_builder import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast


with open("datasets.txt", 'r', encoding='utf-8') as fp:
    datasets = []
    for line  in fp.readlines():
        datasets.append(line.strip('\n'))

# Mettre none dans la troisième valeur du tuple au lieu du dico si aucun paramètre va 
# etre préciser pour le fit. 


# Check if a list argument is provided
if len(sys.argv) > 2:
    # Extract the function name and arguments
    function_name = sys.argv[1]
    args = sys.argv[2] 
    model_name = sys.argv[3]

    # Check if the function exists
    if function_name in globals() and callable(globals()[function_name]):
        # Call the specified function with the provided arguments
        params_model,model_type,params_fit = ast.literal_eval(args)
        if params_model is not None:
            model = (globals()[function_name](**params_model), model_type, params_fit, model_name)
        else:
            model = (globals()[function_name](), model_type, params_fit, model_name)
    else:
        print(f"Function '{function_name}' not found or not callable.")
else:
    print("Insufficient arguments.")

model_list =[model]

## A l'interieur de la boucle sur les données nous devons redéfinir le modèle à chaque itération
## Car sinon le modèle entrainé est sauvegardé en mémoire est il est utilisé lors de la prochaine itération.  

for dataset in datasets:
    if function_name in globals() and callable(globals()[function_name]):
        # Call the specified function with the provided arguments
        params_model,model_type,params_fit = ast.literal_eval(args)
        if params_model is not None:
            model = (globals()[function_name](**params_model), model_type, params_fit, model_name)
        else:
            model = (globals()[function_name](), model_type, params_fit, model_name)
    else:
        print(f"Function '{function_name}' not found or not callable.")
    model_list =[model]
    test_models(model_list=model_list, dataset_name=dataset)
