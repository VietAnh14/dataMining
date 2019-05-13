import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle


def format_data():
    # Pre process data
    df = pd.read_csv('./data/student-mat.csv', sep=';')
    df = df[~df['G3'].isin([0, 1])]
    df = df.rename(columns={'G3': 'Grade'})
    # One-Hot Encoding of Categorical Variables
    df = pd.get_dummies(df)

    # Find correlations with the Grade
    most_correlated = df.corr().abs()['Grade'].sort_values(ascending=False)

    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:8]

    df = df.loc[:, most_correlated.index]
    df = df.drop(columns='schoolsup_no')

    # Save the csv file to train model
    df.to_csv('./data/formatData.csv', index=False)


def split_data(data, train_size):
    test_size = 1 - train_size
    labels = data['Grade']
    data = data.drop(columns='Grade')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train = X_train.rename(columns={'schoolsup_yes': 'schoolsup',
                                      'Medu': 'mother_edu'})
    X_test = X_test.rename(columns={'schoolsup_yes': 'schoolsup',
                                    'Medu': 'mother_edu'})
    return X_train, X_test, y_train, y_test


# Calculate mae and rmse
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    return mae, rmse


# Evaluate several ml models by training on training set and testing on testing set
def train(train_size=0.7):
    data = pd.read_csv('./data/formatData.csv')
    X_train, X_test, y_train, y_test = split_data(data, train_size)
    model_name_list = ['Linear Regression', 'Random Forest']
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(n_estimators=50)

    for model in [linear_model, random_forest_model]:
        model.fit(X_train, y_train)

    pickle.dump(linear_model, open('./trainedModel/linear_model.sav', 'wb'))
    pickle.dump(random_forest_model, open('./trainedModel/random_forest_model.sav', 'wb'))
    # Save test data for accuracy
    X_test.to_csv('./data/x_test.csv', index=False, header=True)
    y_test.to_csv('./data/y_test.csv', index=False, header=True)
    results = dict()
    results[model_name_list[0]] = linear_model.score(X_test, y_test)
    results[model_name_list[1]] = random_forest_model.score(X_test, y_test)
    return results


def predict_grade(sample, op='linear'):
    predict = None
    if (op == 'linear'):
        linear_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
        predict = linear_model.predict(sample)
    elif (op == 'random'):
        random_forest_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
        predict = random_forest_model.predict(sample)
    return predict


# Calculate mae and rmse
def evaluate_metrics():
    # get model and test data
    lr_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    rf_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv').values
    y_test = pd.read_csv('./data/y_test.csv')['Grade'].values
    model_name = ['lr', 'rf']
    results = {}
    for i, model in enumerate([lr_model, rf_model]):
        predictions = model.predict(x_test)
        # Metrics
        metrics = {}
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        metrics['mae'] = mae
        metrics['rmse'] = rmse

        # Insert metrics in to results
        results[model_name[i]] = metrics
    return results


def accuracy():
    accur = {}
    linear_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    random_forest_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv', index_col=False)
    y_test = pd.read_csv('./data/y_test.csv', index_col=False)
    accur['Linear Regression'] = linear_model.score(x_test, y_test)
    accur['Random Forest'] = random_forest_model.score(x_test, y_test)
    return accur


# Linear regression formula
def formula():
    lr_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv')
    lr_formula = 'Grade = %0.2f +' % lr_model.intercept_
    for i, col in enumerate(x_test.columns):
        lr_formula += ' %0.2f * %s +' % (lr_model.coef_[i], col)

    lr_formula = ' '.join(lr_formula.split(' ')[:-1])
    return lr_formula

if __name__ == '__main__':
    # # format_data()
    # df = pd.read_csv('./data/formatData.csv')
    # train(0.75)
    # sample = [[12, 13, 0, 0, 0, 4]]
    # grade = predict_grade(sample, 'linear')
    # # print(grade)
    # print(accuracy())
    # df = pd.read_csv('./data/x_test.csv')
    # a = df.values
    # print(a)
    # a = evaluate_metrics()
    # print(a)
    print(formula())
