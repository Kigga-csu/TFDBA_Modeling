import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_DCDC_data(file='data_fake.xls'):
    # 读取xlsx文件
    data = pd.read_excel(file)

    # 提取输入和输出数据
    X = data.iloc[:, 0:3].values
    y = data.iloc[:, 3].values

    # 数据分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 保存标准化模型
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # 将X_train和Y_train组合为训练集，保存为xls文件
    train_data = pd.DataFrame(X_train)
    train_data['target'] = Y_train
    train_data.to_csv('train_experiment_data.csv', index=False)

    # 将X_test和Y_test组合为测试集，保存为xls文件
    test_data = pd.DataFrame(X_test)
    test_data['target'] = Y_test
    test_data.to_csv('test_experiment_data.csv', index=False)

    return X_train, X_test, Y_train, Y_test

load_DCDC_data('data_real.xls')