import os

import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import joblib


def load_data(input_dir):

    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))

    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))

    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))

    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

    

    # Supprimer la colonne 'date'

    X_train = X_train.drop('date', axis=1)

    X_test = X_test.drop('date', axis=1)

    

    return X_train, X_test, y_train, y_test


def grid_search_models(X_train, y_train, output_dir):

    # Créer les dossiers si nécessaire

    os.makedirs(output_dir, exist_ok=True)

    

    # Définir les modèles et leurs grilles de paramètres

    models = {

        'Ridge': {

            'model': Pipeline([

                ('scaler', StandardScaler()),

                ('regressor', Ridge())

            ]),

            'params': {

                'regressor__alpha': [0.1, 1.0, 10.0],

                'regressor__solver': ['auto', 'svd', 'cholesky']

            }

        },

        'Lasso': {

            'model': Pipeline([

                ('scaler', StandardScaler()),

                ('regressor', Lasso())

            ]),

            'params': {

                'regressor__alpha': [0.1, 1.0, 10.0],

                'regressor__max_iter': [1000, 5000]

            }

        },

        'RandomForest': {

            'model': Pipeline([

                ('scaler', StandardScaler()),

                ('regressor', RandomForestRegressor(random_state=42))

            ]),

            'params': {

                'regressor__n_estimators': [50, 100, 200],

                'regressor__max_depth': [None, 10, 20]

            }

        }

    }

    

    best_models = {}

    

    for name, model_info in models.items():

        # GridSearchCV

        grid_search = GridSearchCV(

            model_info['model'], 

            model_info['params'], 

            cv=5, 

            scoring='neg_mean_squared_error'

        )

        

        # Ajustement

        grid_search.fit(X_train, y_train.values.ravel())

        

        # Sauvegarder les meilleurs paramètres

        best_models[name] = {

            'best_params': grid_search.best_params_,

            'best_score': -grid_search.best_score_  # Conversion du score MSE

        }

        

        # Sauvegarder le modèle

        joblib.dump(grid_search.best_estimator_, 

                    os.path.join(output_dir, f'{name}_best_model.pkl'))

    

    # Sauvegarder les résultats

    import json

    with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:

        json.dump(best_models, f, indent=2)

    

    print("Recherche des meilleurs hyperparamètres terminée.")

    return best_models


def main():

    input_dir = "data/processed"

    output_dir = "models"

    

    # Charger les données

    X_train, X_test, y_train, y_test = load_data(input_dir)

    

    # Recherche des meilleurs hyperparamètres

    best_models = grid_search_models(X_train, y_train, output_dir)

    

    # Afficher les résultats

    for model_name, results in best_models.items():

        print(f"\n{model_name}:")

        print(f"  Meilleurs paramètres : {results['best_params']}")

        print(f"  Meilleur score (MSE) : {results['best_score']}")


if __name__ == "__main__":

    main()
