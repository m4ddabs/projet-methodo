import matplotlib.pyplot as plt
import json
import os

# Get a list of all files in the directory
pth = os.path.join("resultats","TwoPatterns")
all_files = os.listdir(pth)
# Filter the list to include only JSON files
json_files = [file for file in all_files if file.endswith('.json')]

results_hist =[]
for file in json_files:
    with open(os.path.join(pth,file), 'r', encoding="utf-8") as fp:
        results_hist.append(json.load(fp))


#### Plots
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0, 0].plot(results_hist[0]["val_accuracy"])
axs[0, 0].set_title('Modele sequentiel simple')
axs[0, 1].plot(results_hist[1]["val_accuracy"])
axs[0, 1].set_title('Modele sequentiel avec 3 hidden layers')
axs[1, 0].plot(results_hist[2]["val_accuracy"])
axs[1, 0].set_title('Modele RNN simple')


for ax in axs.flat:
    ax.set(xlabel='Epochs', ylabel="Validation Accuracy")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()