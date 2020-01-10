import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# https://www.tensorflow.org/tutorials/estimator/linear

from sklearn.model_selection import train_test_split


def predict_churn():
    df = read_data()
    # explore_data(df)
    df = clean_data(df)
    # visualise_data(df)

    X_train, X_test, y_train, y_test = train_test(df)

    feature_columns = make_feature_columns(X_train)

    classifier = train(feature_columns, X_train, y_train)
    test(classifier, X_test, y_test)


def read_data():
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')

    return df


def explore_data(df):
    """
        Print some basic details about the data set, see what we are working with.
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


def clean_data(df):
    # print(df.info())
    """
        CustomerID is not useful for classification so drop it.
        Boolean features are Yes/No(object) apart from SeniorCitizen, so lets make them consistent.
        TotalCharges should be float, the same as monthly charges.
        Make target feature Churn 1/0
    """
    df.drop('customerID', axis=1, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
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


def visualise_data(df):
    df['Churn'].value_counts().plot(kind='bar', x='Churn', y='count', title='Pos/Neg examples')
    plt.show()


def train_test(df):
    """
        Independent features X, Dependent feature y.
        Reserve 30% of the data for testing and the remaining 70% for training.
        Could plot a learning curve here to evaluate the effect of test_size on accuracy.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


def make_feature_columns(training_data):
    """
        Feature columns describe how the model should interpret the raw input features.
        Here we only use the base features, no derived features. Categorical features
        mapped with their categories. Numeric features as a tensor flow float.
    """
    feature_columns = []

    for feature in training_data:
        if training_data[feature].dtype.name == 'object':
            vocabulary = training_data[feature].unique()
            feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))
        else:
            feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

    return feature_columns


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    """
        Input function converts our dataframe into a tensor flow DataSet
    """
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset

    return input_function


def train(feature_columns, X_train, y_train):
    """
        With our training input function and feature columns we can finally create
        the linear classifier and train it.
    """
    train_input_fn = make_input_fn(X_train, y_train)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)

    return linear_est


def test(linear_est, X_test, y_test):
    """
        Now we unlock the testing data and evaluate our model.
    """
    eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)
    print(linear_est.evaluate(eval_input_fn))


if __name__ == '__main__':
    print('Predicting Customer Churn')
    predict_churn()
