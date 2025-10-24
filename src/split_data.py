import os

import pandas as pd

import requests

from sklearn.model_selection import train_test_split


def download_data(url, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    

    response = requests.get(url)

    response.raise_for_status()

    

    with open(save_path, 'wb') as f:

        f.write(response.content)

    

    print(f"Données téléchargées avec succès dans {save_path}")


def split_data(input_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    

    # Charger les données

    data = pd.read_csv(input_path)

    

    # Séparer features et target

    X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière

    y = data.iloc[:, -1]   # Dernière colonne (silica_concentrate)

    

    # Split des données

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Sauvegarder les données

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)

    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)

    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)

    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    

    print("Division des données terminée.")


def main():

    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

    raw_data_path = "data/raw/raw.csv"

    processed_data_dir = "data/processed"

    

    # Télécharger les données

    download_data(url, raw_data_path)

    

    # Diviser les données

    split_data(raw_data_path, processed_data_dir)


if __name__ == "__main__":

    main()

