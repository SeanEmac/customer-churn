import pandas as pd
import matplotlib.pyplot as plt


def predict_churn():
    df = read_data()
    df = clean_data(df)

    explore_data(df)
    visualise_data(df)


def read_data():
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')

    return df


def explore_data(df):
    """
        Print some basic details about the data set, see what we are
        working with.
    """
    rows, columns = df.shape
    features = list(df.columns)
    target = features.pop()

    print('\nData set is {} rows by {} columns'.format(rows, columns))
    print("Target Feature: {}: {}".format(target, df[target].unique()))
    print("Number of Features:", len(features))

    # Print out a summary of the features
    for feature in features:
        if df[feature].dtype.name == 'object':
            print('\t- {}: {}'.format(feature, df[feature].unique()))
        else:
            print('\t- {}: Continuous'.format(feature))


def visualise_data(df):
    df['Churn'].value_counts().plot(kind='bar', x='Churn', y='count', title='Pos/Neg examples')
    plt.show()


def clean_data(df):
    # print(df.info())
    """
        CustomerID is not useful for classification so drop it.
        Boolean features are Yes/No(object) apart from SeniorCitizen, so lets make them consistent.
        TotalCharges should be float, the same as monthly charges.
    """
    df.drop('customerID', axis=1, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # print(df.isnull().sum())
    """
        11 missing values from TotalCharges, we can either drop these rows
        or compute the value MonthlyCharges * tenure. On second look, that computation is wrong,
        perhaps monthly charges vary month to month. Lets just drop these rows and
        return the cleaned data set.
    """
    df = df.dropna()
    return df


if __name__ == '__main__':
    print('Predicting Customer Churn')
    predict_churn()
