import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import joblib


def normalize_data(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    

    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))

    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))

    

    # Exclure la colonne 'date' de la normalisation

    date_train = X_train['date']

    date_test = X_test['date']

    

    # Préparer les données numériques

    X_train_numeric = X_train.drop('date', axis=1)

    X_test_numeric = X_test.drop('date', axis=1)

    

    # Créer et ajuster le scaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_numeric)

    X_test_scaled = scaler.transform(X_test_numeric)

    

    # Convertir en DataFrame

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)

    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

    

    # Réajouter la colonne 'date'

    X_train_scaled.insert(0, 'date', date_train)

    X_test_scaled.insert(0, 'date', date_test)

    

    # Sauvegarder les données normalisées

    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)

    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)

    

    # Sauvegarder le scaler pour une utilisation ultérieure

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    

    print("Normalisation des données terminée.")


def main():

    input_dir = "data/processed"

    output_dir = "data/processed"

    normalize_data(input_dir, output_dir)


if __name__ == "__main__":

    main()
