import os

import pandas as pd

import numpy as np

import joblib

from sklearn.metrics import mean_squared_error, r2_score


def load_data(input_dir):

    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))

    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))

    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))

    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

    

    # Supprimer la colonne 'date'

    X_train = X_train.drop('date', axis=1)

    X_test = X_test.drop('date', axis=1)

    

    return X_train, X_test, y_train, y_test


def train_model(input_dir, output_dir):

    # Créer les dossiers si nécessaire

    os.makedirs(output_dir, exist_ok=True)

    

    # Charger les données

    X_train, X_test, y_train, y_test = load_data(input_dir)

    

    # Charger le meilleur modèle RandomForest

    best_model = joblib.load('models/RandomForest_best_model.pkl')

    

    # Entraînement final

    best_model.fit(X_train, y_train.values.ravel())

    

    # Prédictions

    y_pred = best_model.predict(X_test)

    

    # Calcul des métriques

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    

    # Sauvegarder le modèle final

    joblib.dump(best_model, os.path.join(output_dir, 'final_model.pkl'))

    

    # Sauvegarder les métriques

    import json

    metrics = {

        'mean_squared_error': mse,

        'r2_score': r2

    }

    

    with open(os.path.join(output_dir, 'train_metrics.json'), 'w') as f:

        json.dump(metrics, f, indent=2)

    

    print(f"Modèle entraîné. Métriques :")

    print(f"  MSE : {mse}")

    print(f"  R2 Score : {r2}")


def main():

    input_dir = "data/processed"

    output_dir = "models"

    train_model(input_dir, output_dir)


if __name__ == "__main__":

    main()
