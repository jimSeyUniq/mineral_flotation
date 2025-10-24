import os

import pandas as pd

import numpy as np

import joblib

from sklearn.metrics import (

    mean_squared_error, 

    mean_absolute_error, 

    r2_score, 

    explained_variance_score

)

import json


def load_data(input_dir):

    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))

    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

    

    # Supprimer la colonne 'date'

    X_test = X_test.drop('date', axis=1)

    

    return X_test, y_test


def evaluate_model(input_dir, output_dir, model_path):

    # Créer les dossiers si nécessaire

    os.makedirs(output_dir, exist_ok=True)

    

    # Charger les données de test

    X_test, y_test = load_data(input_dir)

    

    # Charger le modèle

    model = joblib.load(model_path)

    

    # Prédictions

    y_pred = model.predict(X_test)

    

    # Calcul des métriques

    metrics = {

        'mean_squared_error': mean_squared_error(y_test, y_pred),

        'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),

        'mean_absolute_error': mean_absolute_error(y_test, y_pred),

        'r2_score': r2_score(y_test, y_pred),

        'explained_variance_score': explained_variance_score(y_test, y_pred)

    }

    

    # Sauvegarder les prédictions

    predictions_df = pd.DataFrame({

        'true_values': y_test['silica_concentrate'],

        'predicted_values': y_pred

    })

    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    

    # Sauvegarder les métriques

    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:

        json.dump(metrics, f, indent=2)

    

    # Afficher les métriques

    print("Métriques d'évaluation :")

    for metric, value in metrics.items():

        print(f"  {metric}: {value}")


def main():

    input_dir = "data/processed"

    output_dir = "metrics"

    model_path = "models/final_model.pkl"

    

    evaluate_model(input_dir, output_dir, model_path)


if __name__ == "__main__":

    main()
