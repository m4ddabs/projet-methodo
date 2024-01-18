import json 
import pandas as pd 
import os



datasets = os.listdir("resultats")



results_dict = {'Data': [], 'Model': [], 'Test_Accuracy': []}

for dataset in datasets:
    pth = os.path.join("resultats", dataset)
    all_files = os.listdir(pth)
    # on récupere les fichiers json 
    json_files = [file for file in all_files if file.endswith('.json')]
    for file in json_files:
        filepath = os.path.join(pth, file)

        with open(filepath, 'r', encoding="utf-8") as fp:
            # Chargement des données JSON
            data = json.load(fp)
            # Récupération des noms de modèles 
            model_name = os.path.splitext(file.strip(".json"))[0]
            # Récupération des noms de modèles 
 
            test_accuracy = data['test_accuracy']
 
            results_dict['Data'].append(dataset)
            results_dict['Model'].append(model_name)
            results_dict['Test_Accuracy'].append(test_accuracy)

df = pd.DataFrame(results_dict)
df_pivoted = df.pivot(index='Data', columns ='Model', values="Test_Accuracy")

print(df_pivoted.head())

df_pivoted.to_csv('R_projet/results.csv', index=True)
