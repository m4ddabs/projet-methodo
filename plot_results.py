import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Get a list of all files in the directory
pth = os.path.join("resultats","TwoPatterns")
all_files = os.listdir(pth)
# Filter the list to include only JSON files
json_files = [file for file in all_files if file.endswith('.json')]

model_names = []
results_hist =[]
for file in json_files:
    with open(os.path.join(pth,file), 'r', encoding="utf-8") as fp:
        results_hist.append(json.load(fp))
        model_names.append(os.path.splitext(file)[0])




fig, axs = plt.subplots(len(results_hist), 2, figsize=(10,4  * len(results_hist)))

for i, result in enumerate(results_hist):
    model_name = model_names[i]  # Nom du modèle correspondant à l'itération actuelle

    # Graphique pour la précision (Accuracy)
    axs[i, 0].plot(range(1, result['epochs'] + 1), result['accuracy'], label='Train Accuracy')
    axs[i, 0].plot(range(1, result['epochs'] + 1), result['val_accuracy'], label='Validation Accuracy')
    axs[i, 0].set_title(f'Model Accuracy - {model_name}')
    axs[i, 0].set_xlabel('Epochs')
    axs[i, 0].set_ylabel('Accuracy')
    axs[i, 0].legend()

    
    min_val_loss_index = result['val_loss'].index(min(result['val_loss']))+1
    axs[i, 0].axvline(x = min_val_loss_index, color='green', linestyle='--', label='Min Loss Epoch')
    axs[i, 0].legend()

    # Graphique pour la perte (Loss)
    axs[i, 1].plot(range(1, result['epochs'] + 1), result['loss'], label='Train Loss')
    axs[i, 1].plot(range(1, result['epochs'] + 1), result['val_loss'], label='Validation Loss')
    axs[i, 1].set_title(f'Model Loss - {model_name}')
    axs[i, 1].set_xlabel('Epochs')
    axs[i, 1].set_ylabel('Loss')
    axs[i, 1].legend()
    

    axs[i, 1].axvline(x = min_val_loss_index, color='green', linestyle='--', label='Min Loss Epoch')
    axs[i, 1].legend()
    
plt.show()
