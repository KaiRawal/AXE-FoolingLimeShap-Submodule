import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

PGI_STD = 0.4
try:
    PGI_STD = float(os.getenv('EVAL_PGI_STD', '0.4'))
except (TypeError, ValueError):
    PGI_STD = 0.4

def get_bex(df, model, explanation_feature, k=3):
    """
    Calculate the BEX score (BlindEvalEx) score for a given feature in the dataset.
    The BEX score is just the accuracy of a knn model fit on dataset (x,y) where x=df[explanation_feature] and y=model(df)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset on which the model is evaluated. It should be a pandas DataFrame.
    model : object
        The model used for prediction. It should have a `predict_proba` method.
    explanation_feature : str
        The feature for which the BEX score is calculated.

    Returns
    ----------
    float
        The BEX score for the specified feature.
    """

    # Extract the feature and the model predictions
    X = df[[explanation_feature]].values
    y = model.predict(df.values)

    # Initialize and train the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Predict on the test set
    y_pred = knn.predict(X)

    # Calculate the accuracy
    bex_score = accuracy_score(y, y_pred)

    return bex_score

def get_pgi(df, model, explanation_feature):
    """
    Calculate the Predictive Global Importance (PGI) of a feature for a given model and dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset on which the model is evaluated. It should be either a pandas DataFrame.
    model : object
        The model used for prediction. It should have a `predict_proba` method.
    explanation_feature : str
        The feature for which the PGI is calculated.

    Returns
    ----------
    float
        The PGI value for the specified feature.
    """
    data = df.copy(deep=True)
    df_inputs = df.copy(deep=True)
    
    data['y_hat'] = model.predict_proba(df_inputs.values)[:, 1]
    
    data['pgi'] = 0
    num_samples = 100
    for i in range(num_samples):
        df_inputs = df.copy(deep=True)
        # df_inputs[explanation_feature] += np.random.normal(0, 0.3, size=len(df_inputs[explanation_feature]))

        # if False and len(df_inputs[explanation_feature].unique()) == 2:
        #     unique_values = df_inputs[explanation_feature].unique()
        #     flip_mask = np.random.rand(len(df_inputs)) < -0.3
        #     df_inputs.loc[flip_mask, explanation_feature] = df_inputs.loc[flip_mask, explanation_feature].apply(
        #         lambda x: unique_values[1] if x == unique_values[0] else unique_values[0]
        #     )
        # else:
        #     df_inputs[explanation_feature] += np.random.normal(0, float(os.getenv('EVAL_PGI_STD')), size=len(df_inputs[explanation_feature]))
        df_inputs[explanation_feature] += np.random.normal(0, PGI_STD, size=len(df_inputs[explanation_feature]))
        
        data[f'fxp{i}'] = model.predict_proba(df_inputs.values)[:, 1]
        data[f'fxp{i}'] -= data['y_hat']
        data[f'fxp{i}'] = abs(data[f'fxp{i}'])
        data['pgi'] += data[f'fxp{i}']
        data.drop(columns=[f'fxp{i}'], inplace=True)
    data['pgi'] /= num_samples

    return data['pgi'].values




def get_pgi_multi(df, model, explanation_features):
    """
    Calculate the Predictive Global Importance (PGI) of a list of features for a given model and dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset on which the model is evaluated. It should be either a pandas DataFrame.
    model : object
        The model used for prediction. It should have a `predict_proba` method.
    explanation_features : list of str
        The list of features for which the PGI is calculated.

    Returns
    ----------
    float
        The PGI value for the specified features.
    """
    data = df.copy(deep=True)
    df_inputs = df.copy(deep=True)
    
    data['y_hat'] = model.predict_proba(df_inputs.values)[:, 1]
    
    data['pgi'] = 0
    num_samples = 100
    for i in range(num_samples):
        df_inputs = df.copy(deep=True)
        for feature in explanation_features:
            df_inputs[feature] += np.random.normal(0, PGI_STD, size=len(df_inputs[feature]))
        data[f'fxp{i}'] = model.predict_proba(df_inputs.values)[:, 1]
        data[f'fxp{i}'] -= data['y_hat']
        data[f'fxp{i}'] = abs(data[f'fxp{i}'])
        data['pgi'] += data[f'fxp{i}']
        data.drop(columns=[f'fxp{i}'], inplace=True)
    data['pgi'] /= num_samples

    return data['pgi'].values

