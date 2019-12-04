# Spark project MS Big Data Télécom : Kickstarter campaigns

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020

Pour le Preprocessor un répertoire data a été crée dans l'arborescence du projet dans lequel on ira lire les données (train_clean.csv).
A la fin du trainer on écrira le dataframe final au format parquet dans ce même répertoire.
Pour le Trainer nous nous sommes basé sur le dataset fourni, nous lisons dans le répertoire data les données prepared_trainingset et nous écrivons les résultats dans le répertoire data (le répertoire LogisticRegression sera crée à l'issue de l'execution du Trainer).
Si un répertoire est déjà présent lorsque nous souhaitons écrire les résultats alors on l'écrase et on stocke les résultats de la dernière execution.

