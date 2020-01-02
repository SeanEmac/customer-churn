import pandas as pd


def read_data():
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')
    df = df.drop(columns=['customerID'])

    print(df.head())
    rows, columns = df.shape
    features = list(df.columns)
    target = features.pop()

    return df
