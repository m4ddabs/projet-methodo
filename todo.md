## Taches à faire pour jeudi 30 novembre: 

* Création d'un dépot github -- Fait
* Création d'un colab -- Fait
* Importation des jeux de données (2 ou 3) -- Fait
* Implémentation stratégie récurrente -- Fait
* Créer des fonctions pour rendre automatique le test de différents modèles
  * Fonction de préparation des données pour chaque type d'architecture de réseau -- Fait
  * Fonction de test des modeles qui retourne l'historique de chaque modele et le modele, et qui sauvegarde les modeles (ses poids, son accuracy etc) dans un fichier txt. -- Fait 
  * faire en sorte de donner un nom pertinent à chaque algo. -- Fait

## Taches à faire pour vendredi 15 décembre: 

* Retravailler la fonction prepare_data et voir que la fonction to_categorical fonctionne avec n'importe quel type de label(convertir tous les label en valeurs entre 0 - n-1 classes) Fait
* Créer un dictionnaire paramètres à passer en entrée avec chaque modèle, pour pouvoir spécifier des paramètres spécifiques pour chaque modèle. Fait  
* Automatiser la création de modèles, faire en sorte qu'on puisse préciser le shape du input ainsi que le nombre de classes et les couches intermédiaire. Une fonction par type d'architercutre par exemple ? Fait
* Séparer la visualation des données de la partie test des modèles.Fait
* Faire en sorte que dans les visu on voit Train et Test séparément. Fait 
* Séparer xtrain et ytrain en un jeu de validation et garder xtest et ytest pour mesurer l'accuracy avec evaluate. Fait
* Faire en sorte que dans le json des performances on garde les valeurs du evaluate aussi et pas juste history. Fait
* Faire un script de synthèse des résultats, les mettre sous forme d'un tableau automatiquement
* Script bash pour lancer plusieurs test simultanément. Une fonction test module par test et par fichier ou une grosse fonction test module par fichier ? 



