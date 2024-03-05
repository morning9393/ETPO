import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.datasets import fetch_olivetti_faces, fetch_rcv1, fetch_lfw_pairs, fetch_lfw_people, fetch_kddcup99, fetch_covtype, fetch_20newsgroups_vectorized
from caafe import data
from caafe.preprocessing import make_datasets_numeric

def load_kidney_stone(split=True):
    data_disc = "Kidney Stone"
    train_data_path = "../../mappo/envs/datascience/kidney_stone/train.csv"
    test_data_path = "../../mappo/envs/datascience/kidney_stone/test.csv"
    # load csv files to sklearn dataset
    train_data = pd.read_csv(train_data_path, index_col=0)
    # test_data = pd.read_csv(test_data_path, index_col=0)
    x = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values
    if split:
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test, data_disc
    else:
        return x, y, data_disc

def load_pharyngitis(split=True):
    data_disc = "Pharyngitis"
    data_path = "../../mappo/envs/datascience/pharyngitis/pharyngitis.csv"
    # load csv files to sklearn dataset
    data = pd.read_csv(data_path)
    data = data.dropna(axis=0, how='any', inplace=False)
    x = data.drop(["number"],axis=1).drop(["radt"],axis=1).values
    y = data["radt"].values
    if split:
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test, data_disc
    else:
        return x, y, data_disc
    
def load_health_insurance(split=True):
    data_disc = "Health Insurance"
    data_path = "../../mappo/envs/datascience/health_insurance/health_insurance.csv"
    # load csv files to sklearn dataset
    data = pd.read_csv(data_path)
    data = data.dropna(axis=0, how='any', inplace=False)
    data = data.values[:2000, 1:]
    for i in range(data.shape[1]):
        # encode categorical features, better to use onehot encoder
        if data[:, i].dtype == object:
            le = LabelEncoder()
            data[:, i] = le.fit_transform(data[:, i])
    x = data[:, :-1]
    y = data[:, -1].astype(np.int32)
    if split:
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test, data_disc
    else:
        return x, y, data_disc
    
def load_spaceship_titanic(split=True):
    data_disc = "Spaceship Titanic"
    data_path = "../../mappo/envs/datascience/spaceship_titanic/spaceship_titanic.csv"
    # load csv files to sklearn dataset
    data = pd.read_csv(data_path)
    data = data.dropna(axis=0, how='any', inplace=False)
    data = data.values[:2000, 1:]
    for i in range(data.shape[1]):
        # encode categorical features, better to use onehot encoder
        if data[:, i].dtype == object:
            le = LabelEncoder()
            data[:, i] = le.fit_transform(data[:, i])
    x = data[:, :-1]
    y = data[:, -1].astype(np.int32)
    if split:
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test, data_disc
    else:
        return x, y, data_disc

def load_caafe_dataset(dataset_name, split=True):
    datasets = data.load_all_data()
    if dataset_name == "balance_scale":
        data_disc = "Balance Scale Weight & Distance"
        ds = datasets[0]
    elif dataset_name == "breast_w":
        data_disc = "Breast Cancer Wisconsin"
        ds = datasets[1]
    elif dataset_name == "cmc":
        data_disc = "National Indonesia Contraceptive Prevalence"
        ds = datasets[2]
    elif dataset_name == "credit_g":
        data_disc = "German Credit"
        ds = datasets[3]
    elif dataset_name == "diabetes":
        data_disc = "Diabetes"
        ds = datasets[4]
    elif dataset_name == "tic_tac_toe":
        data_disc = "Tic-Tac-Toe Endgame"
        ds = datasets[5]
    elif dataset_name == "eucalyptus":
        data_disc = "Eucalyptus Soil Conservation"
        ds = datasets[6]
    elif dataset_name == "pc1":
        data_disc = "PC1 Software Defect"
        ds = datasets[7]
    elif dataset_name == "airlines":
        data_disc = "Airlines"
        ds = datasets[8]
    elif dataset_name == "jungle_chess":
        data_disc = "Jungle Chess"
        ds = datasets[9]
    else:
        raise NotImplementedError
    
    target_column_name = ds[4][-1]
    ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
    ds = pd.concat([df_train, df_test])
    ds = ds.dropna(axis=0, how='any', inplace=False)
    x, y = data.get_X_y(ds, target_column_name)
    x = x.numpy()
    y = y.numpy()
    
    # x = ds[1]
    # y = ds[2]
    if split:
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test, data_disc
    else:
        return x, y, data_disc

def load_preinstalled_dataset():
    print("load_preinstalled_dataset")
    dataset = load_digits()
    if dataset.target.dtype == object:
            le = LabelEncoder()
            dataset.target = le.fit_transform(dataset.target)
    for i in range(dataset.data.shape[1]):
        # encode categorical features, better to use onehot encoder
        if dataset.data[:, i].dtype == object:
            le = LabelEncoder()
            dataset.data[:, i] = le.fit_transform(dataset.data[:, i])

    print("dataset.data.shape: ", dataset.data.shape)
    print("dataset.data.type: ", type(dataset.data))
    x_train, x_test, y_train, y_test = \
            train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    print("x train shape: ", x_train.shape)
    print("x train type: ", type(x_train))
    print("y train shape: ", y_train.shape)
    return x_train, x_test, y_train, y_test