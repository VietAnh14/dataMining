import pandas as pd
import numpy as np
# Import model cần sử dụng từ thư viện sk learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle


# Hàm tiền xử lý dữ liệu
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
    # Lấy 6 thuộc tính, ở đây lấy 8 vì bao gồm cả thuộc tính Grade và Schoolsup_no
    most_correlated = most_correlated[:8]

    df = df.loc[:, most_correlated.index]

    # Bỏ thuộc tính không cần thiết là schoolsup_no
    df = df.drop(columns='schoolsup_no')

    # Save the csv file to train model
    df.to_csv('./data/formatData.csv', index=False)


# Hàm tách dữ liệu thành test data và train data
# Nhận vào train size
def split_data(data, train_size):
    test_size = 1 - train_size

    # Thuộc tính target
    labels = data['Grade']

    # Bỏ thuộc tính Grade khỏi data
    data = data.drop(columns='Grade')

    # Tách data và labels ra train và test data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    # Đổi tên các cột
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


# Train model dựa trên train size, giá trị mặc định train size là 0.7
def train(train_size=0.7):
    # Đọc data
    data = pd.read_csv('./data/formatData.csv')

    # Tách data thành train và test
    X_train, X_test, y_train, y_test = split_data(data, train_size)
    model_name_list = ['Linear Regression', 'Random Forest']

    # Khởi tạo 2 model, random forest với số cây là 50
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(n_estimators=50)

    # Train model
    for model in [linear_model, random_forest_model]:
        model.fit(X_train, y_train)

    # Lưu model lại để sử dụng sau
    pickle.dump(linear_model, open('./trainedModel/linear_model.sav', 'wb'))
    pickle.dump(random_forest_model, open('./trainedModel/random_forest_model.sav', 'wb'))

    # Save test data for accuracy
    X_test.to_csv('./data/x_test.csv', index=False, header=True)
    y_test.to_csv('./data/y_test.csv', index=False, header=True)
    results = dict()

    # Tính R - Square của 2 model
    results[model_name_list[0]] = linear_model.score(X_test, y_test)
    results[model_name_list[1]] = random_forest_model.score(X_test, y_test)
    return results


# Dự đoán điểm dựa trên mẫu đưa vào, co thể chọn mô hình để dự đoán, mặc định là linear model
def predict_grade(sample, op='linear'):
    predict = None
    if (op == 'linear'):
        # Load model
        linear_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
        predict = linear_model.predict(sample)
    elif (op == 'random'):
        random_forest_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
        predict = random_forest_model.predict(sample)
    return predict


# Calculate mae and rmse
# Hàm tính giá trị mse( mean absolute error) và rmse (root mean square error)
def evaluate_metrics():
    # get model and test data
    lr_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    rf_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv').values
    y_test = pd.read_csv('./data/y_test.csv')['Grade'].values
    # tạo mảng tên model
    model_name = ['lr', 'rf']
    results = {}
    # duyệt qua chỉ số index và tên model trong mảng model
    for i, model in enumerate([lr_model, rf_model]):
        predictions = model.predict(x_test)
        # Metrics
        metrics = {}

        # Tính giá trị mae và rmse của model theo công thức
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        metrics['mae'] = mae
        metrics['rmse'] = rmse

        # Thêm số liệu vào results
        # Insert metrics in to results
        results[model_name[i]] = metrics
    return results


# Hàm tìm giá trị R- Square (R^2) của model
def r_square():
    r2 = {}
    linear_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    random_forest_model = pickle.load(open('./trainedModel/random_forest_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv', index_col=False)
    y_test = pd.read_csv('./data/y_test.csv', index_col=False)
    # Tính R - Square của 2 model
    r2['Linear Regression'] = linear_model.score(x_test, y_test)
    r2['Random Forest'] = random_forest_model.score(x_test, y_test)
    return r2


# Linear regression formula
# Tìm phương trình hồi quy
def formula():

    # Load model và test data được lưu lên để tìm hàm số
    lr_model = pickle.load(open('./trainedModel/linear_model.sav', 'rb'))
    x_test = pd.read_csv('./data/x_test.csv')

    # Phương trình hồi quy: Grade = w0 + w1*X1 + w2*x2 +....
    # lr_model.intercept_ hằng số độc lập trong mô hình tuyến tính (w0)
    # lr_model.coef_mảng chứa hệ số mối quan hệ của mô hình tuyến tính (w1, w2,...)
    lr_formula = 'Grade = %0.2f +' % lr_model.intercept_
    for i, col in enumerate(x_test.columns):
        lr_formula += ' %0.2f * %s +' % (lr_model.coef_[i], col)

    # Loại bỏ dấu '+' cuối cùng.
    lr_formula = ' '.join(lr_formula.split(' ')[:-1])
    return lr_formula

# if __name__ == '__main__':      # test
#     # # format_data()
#     #     # df = pd.read_csv('./data/formatData.csv')
#     #     # train(0.75)
#     #     # sample = [[12, 13, 0, 0, 0, 4]]
#     #     # grade = predict_grade(sample, 'linear')
#     #     # # print(grade)
#     #     # print(accuracy())
#     #     # df = pd.read_csv('./data/x_test.csv')
#     #     # a = df.values
#     #     # print(a)
#     #     # a = evaluate_metrics()
#     #     # print(a)
#     print(evaluate_metrics())
#     print(formula())
