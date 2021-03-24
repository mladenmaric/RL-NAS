import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from preprocessing.config import DATASET


def preprocess_data():
    if DATASET == 'cancer':
        return preprocess_data_cancer()
    elif DATASET == 'churn':
        return preprocess_data_churn()
    elif DATASET == 'mexie':
        return preprocess_data_mexie()
    elif DATASET == 'boston':
        return preprocess_data_boston()
    elif DATASET == 'real_estate':
        return preprocess_data_real_estate()


def preprocess_data_cancer():
    df = pd.read_csv(os.path.join('datasets', 'cancer', 'breast-cancer-wisconsin.data'), header=None, na_values='?')

    scaler = MinMaxScaler()
    labeler = LabelEncoder()

    X = df.iloc[:, 1:-1]
    X = X.fillna(X.median())
    y = df.iloc[:, -1]

    X = np.array(scaler.fit_transform(X))
    y = np.array(labeler.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=7777)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def preprocess_data_churn():
    df = pd.read_csv(os.path.join('datasets', 'churn', 'Churn_Modelling.csv'))

    y = np.array(df['Exited'])

    geography_dummies = pd.get_dummies(df['Geography'])
    gender_dummies = pd.get_dummies(df['Gender'])
    df = df.drop(columns=['Geography', 'Gender', 'Exited'])
    df = df.join(geography_dummies)
    df = df.join(gender_dummies)

    df['Age'] = df['Age'].transform(lambda x: int(x) // 10)
    df['Balance'] = df['Balance'].transform(lambda x: int(x) // 10000)
    df['CreditScore'] = df['CreditScore'].transform(lambda x: int(x) // 100)

    min_max_scaler = MinMaxScaler()
    columns_scaler = ['Age', 'Balance', 'CreditScore', 'Tenure', 'NumOfProducts', 'EstimatedSalary']
    df[columns_scaler] = min_max_scaler.fit_transform(df[columns_scaler])

    X = np.array(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=7777)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def preprocess_data_mexie():
    feature_columns = [1, 2, 3, 4, 5, 6]
    target_columns = [7, 8]
    data = pd.read_csv(os.path.join('datasets', 'mexie', "dataMexie.csv"))
    data_noiter = pd.read_csv(os.path.join('datasets', 'mexie', "dataMexieNoIter.csv"))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled_noiter = scaler.transform(data_noiter)

    training_data = data_scaled[data['testid'].isin(range(1, 18, 1))]

    X = training_data[:, feature_columns]
    Y = training_data[:, target_columns]

    val_data = data_scaled_noiter[data_noiter['testid'].isin(range(18, 20, 1))]
    X_val = val_data[:, feature_columns]
    Y_val = val_data[:, target_columns]

    return {
        'X_train': X,
        'y_train': Y,
        'X_val': X_val,
        'y_val': Y_val,
        'X_test': X_val,
        'y_test': Y_val
    }


def preprocess_data_boston():
    col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv(os.path.join('datasets', 'boston', 'housing.csv'), delim_whitespace=True, names=col)
    df1 = df[['RM', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']]
    df2 = df1[~(df1['MEDV'] == 50)]

    TAX_10 = df2[(df2['TAX'] < 600) & (df2['LSTAT'] >= 0) & (df2['LSTAT'] < 10)]['TAX'].mean()
    TAX_20 = df2[(df2['TAX'] < 600) & (df2['LSTAT'] >= 10) & (df2['LSTAT'] < 20)]['TAX'].mean()
    TAX_30 = df2[(df2['TAX'] < 600) & (df2['LSTAT'] >= 20) & (df2['LSTAT'] < 30)]['TAX'].mean()
    TAX_40 = df2[(df2['TAX'] < 600) & (df2['LSTAT'] >= 30)]['TAX'].mean()

    indexes = list(df2.index)
    for i in indexes:
        if df2['TAX'][i] > 600:
            if 0 <= df2['LSTAT'][i] < 10:
                df2.at[i, 'TAX'] = TAX_10
            elif 10 <= df2['LSTAT'][i] < 20:
                df2.at[i, 'TAX'] = TAX_20
            elif 20 <= df2['LSTAT'][i] < 30:
                df2.at[i, 'TAX'] = TAX_30
            elif df2['LSTAT'][i] > 30:
                df2.at[i, 'TAX'] = TAX_40

    df3 = df2.drop(axis=0, index=[365, 367])
    df3 = df3.drop(axis=0, index=[364])

    X = df3.iloc[:, 0:4].values
    y = df3.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=987)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def preprocess_data_real_estate():
    df = pd.read_csv(os.path.join('datasets', 'real_estate', 'Real estate.csv'))

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    X = df[:, 1:-1]
    y = df[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=7777)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
