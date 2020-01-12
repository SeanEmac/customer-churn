from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

"""
    I have done some ML tasks like this in College but only with scikit-learn, I have never used TensorFlow before.
    Feature columns and input functions were confusing at first but I followed some guides in the docs
    and eventually got the hang of it, mainly https://www.tensorflow.org/tutorials/estimator/linear and
    lots of Stack Overflow.
"""


def predict_churn():
    df = read_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    explore_data(df)
    df = clean_data(df)
    # visualise_data(df)

    X_train, X_test, y_train, y_test = train_test(df)
    feature_columns = make_feature_columns(X_train)

    classifier = train(feature_columns, X_train, y_train)
    predictions = predict(classifier, X_test)
    evaluate(classifier, predictions, X_test, y_test)
    # I also tested the model against the training data and had similar accuracy levels
    # This is a positive sign that the model is not over/under fitting


def read_data(filename):
    """
        I'm not too familiar with error handling in python but this seems like an easy
        place for the program to fail if someone was just copying this file and not the full repo..
    """
    try:
        df = pd.read_csv(filename, sep=',')
        return df

    except Exception:
        print("Error reading input CSV", filename)
        raise


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
            print('- {}: {}'.format(feature, df[feature].unique()))
        else:
            print('- {}: Continuous'.format(feature))

    # Senior Citizen & Total Charges are incorrectly typed


def clean_data(df):
    """
        CustomerID is not useful for classification so drop it.
        Boolean features are Yes/No(object) apart from SeniorCitizen, so lets make them consistent.
        TotalCharges should be float, the same as monthly charges.
        Make target feature Churn 1/0 because tensorFlow expects label_vocabulary to already be encoded.
    """
    # print('\nDataframe types:')
    # print(df.info())
    df.drop('customerID', axis=1, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # print('\nMissing values:')
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
    """
        I won't plot every feature, just have a look at some I think might be interesting,
        I will comment any insights gained from the visualisations.
    """
    # Plot some features on their own to look at distribution
    # 26.58% of customers churn, slight imbalance but it's not too extreme
    build_plot(df, 'Churn', 'Customer Churn')
    # 16.24% of customers are senior citizens, they could be less likely to churn? Will check later.
    build_plot(df, 'SeniorCitizen', 'Senior citizen')
    # 50.5% males
    build_plot(df, 'gender', 'Gender')
    # 8.72% of customers have only joined in the last month, lots of new customers.
    build_plot(df, 'tenure', 'Tenure')
    # 55% are on a month to month contract, high risk, could churn at any time.
    build_plot(df, 'Contract', 'Contract')

    # Compare some features and their effect on churn
    # Senior citizens are actually more likely to churn, this was surprising
    build_plot(df, 'SeniorCitizen', 'Churn in Senior citizens', stacked=True)
    # As I suspected, month to month contracts are much more likely to churn
    build_plot(df, 'Contract', 'Churn by contract', stacked=True)
    # 62% of new customers are leaving after the first month, this is a bad sign
    build_plot(df, 'tenure', 'Churn by tenure', stacked=True)


def build_plot(data, col1, title, stacked=False):
    """
        We can either build Singe bar charts or stacked bar charts.
        Single: each bar represents a value from that column expressed as a % of the total.
        Stacked: shows the effect of a features value on churn, again as a % of total, so that for example
        we can compare the churn % in Males vs Females.
    """
    if stacked:
        axis = data.groupby([col1, 'Churn']).size().unstack().apply(lambda x: (x / x.sum()) * 100, axis=1).plot(
            kind='bar', title=title, stacked=True)
    else:
        axis = data[col1].value_counts(normalize=True).apply(lambda x: x * 100).plot(
            kind='bar', title=title)

    # so the text doesn't overlap the line
    spacing: int = 1
    for bar in axis.patches:
        height = bar.get_height()
        y_adjust = bar.get_y()
        # This was a bit tricky but it shows the exact % on top of the bar, adjusted if it is stacked
        axis.text(bar.get_x() + bar.get_width() / 2, height + y_adjust + spacing,
                  '{0:.2f}%'.format(height), ha='center', va="center")

    # % ticks on the y axis
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()


def train_test(df):
    """
        Independent features X, Dependent feature y.
        Reserve 30% of the data for testing and the remaining 70% for training.
        Later on I might use 10 fold cross validation for a more accurate score.
        Could plot a learning curve to evaluate the effect of test_size on accuracy.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def make_feature_columns(training_data):
    """
        Feature columns describe how the model should interpret the raw input features.
        Here we only use the base features, no derived features.
        categorical_column_with_vocabulary in a way, changes the String values in our dataframe
        into a discrete set of categorical values. Numeric_column for continuous values.
    """
    feature_columns = []
    for feature in training_data:
        if training_data[feature].dtype.name == 'object':
            vocabulary = training_data[feature].unique()
            feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))
        else:
            feature_columns.append(tf.feature_column.numeric_column(feature))

    return feature_columns


def train(feature_columns, X_train, y_train):
    """
        With our input function and feature columns done we can finally create our model.
        This is a binary classification task, I will use a Linear Classifier.
    """
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X_train, y=y_train, batch_size=128, num_epochs=1000, shuffle=True)

    # tried some of the available optimizers and Ftrl seems to perform well.
    linear_est = tf.estimator.LinearClassifier(optimizer='Ftrl', feature_columns=feature_columns)
    linear_est.train(train_input_fn, steps=1000)

    return linear_est


def predict(linear_est, X_test):
    """
        Now our model has been trained, we can use it to make some predictions
        on the test data that we have kept separate up until now.
    """
    prediction_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X_test, batch_size=128, num_epochs=1, shuffle=False)

    return list(linear_est.predict(prediction_input_func))


def evaluate(linear_est, predictions, X_test, y_test):
    """
        Now we unlock the testing data and evaluate our model.
        Since I used a linear classifier, I can evaluate it with a
        confusion matrix, accuracy scores and a ROC curve since it is probabilistic.
        Accuracy comes out in the high 70s, which is ok, but considering the imbalance
        in the dataset(73.42% No Churn) I would have liked to be able to improve it.
    """
    class_predictions = pd.Series([pred['class_ids'][0] for pred in predictions])
    probabilities = pd.Series([pred['probabilities'][1] for pred in predictions])

    print('Confusion matrix:')
    print(tf.math.confusion_matrix(
        y_test,
        class_predictions,
        num_classes=2
    ))

    print('Evaluation')
    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
    print(linear_est.evaluate(test_input_fn))

    probabilities.plot(kind='hist', bins=20, title='predicted probabilities')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()


if __name__ == '__main__':
    predict_churn()
