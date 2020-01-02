from src import preprocessing as pre


def predict_churn():
    df = pre.read_data()
    print("Starting:")


if __name__ == '__main__':
    print('Predicting customer churn')
    predict_churn()