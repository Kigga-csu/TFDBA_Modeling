import pandas as pd
#以此类推下载所有的软件包 就是import 后面的
import os
import sklearn
from sklearn.model_selection import KFold,train_test_split#这三种只需要下载一个sk learn 但是名字我忘记了你自己去搜
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
from BP_NN import build_model

def load_DCDC_data (file='data/data_fake.xls'):
    # 读取xlsx文件
    data = pd.read_excel(file)

    # 提取输入和输出数据
    X = data.iloc[:, 0:3].values
    y = data.iloc[:, 3].values
    # 数据分割
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    #print(X_train)
    X_train = scaler.fit_transform(X_train)
    #print(X_train)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    return X_train, X_test, Y_train, y_test

X_train, X_test, Y_train, y_test = load_DCDC_data()

# 创建1个文件夹，保存神经网络模型
file_path = os.getcwd()
file_path = file_path+"\model_paramater"
if not os.path.exists(file_path):
    os.makedirs(file_path)

#创建2个空表格，保存实验结果
results1 = pd.DataFrame(columns=['Fold', 'Valid_MSE_NN', 'Valid_MAE_NN', 'Valid_R2_NN', 'Valid_MSE_XGB', 'Valid_MAE_XGB', 'Valid_R2_XGB','Valid_MSE_RF', 'Valid_MAE_RF', 'Valid_R2_RF','Valid_MSE_LR', 'Valid_MAE_LR', 'Valid_R2_LR', 'Test_MSE_NN', 'Test_MAE_NN', 'Test_R2_NN','Test_MSE_XGB', 'Test_MAE_XGB', 'Test_R2_XGB','Test_MSE_RF', 'Test_MAE_RF', 'Test_R2_RF','Test_MSE_LR', 'Test_MAE_LR', 'Test_R2_LR'])
results2 = pd.DataFrame(columns=['Fold', 'Valid_percentage_NN',  'Valid_percentage_XGB', 'Valid_percentage_RF', 'Valid_percentage_LR', 'Test_percentage_NN', 'Test_percentage_XGB', 'Test_percentage_RF','Test_percentage_LR', ])

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# 创建一个空列表，用于存储每一折（fold）的损失值
fold_losses = []
# 创建一个空表格，可能用于存储最终的预测结果
final = []
f = pd.DataFrame(final)

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Training fold {fold + 1}")
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    # 4种模型的构建
    model = build_model()
    model.compile(optimizer='adam', loss='mean_squared_error')

    modelxgb = xgb.XGBRegressor()
    modelRF = RandomForestRegressor()
    modelLR = sklearn.linear_model.LinearRegression()

    # 模型的训练
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=20000, batch_size=256, verbose=1)
    modelxgb.fit(x_train, y_train)
    modelRF.fit(x_train, y_train)
    modelLR.fit(x_train, y_train)

    # 模型性能测试
    # val集合上的性能结果
    Y_NN_pred = model.predict(x_val)
    Y_XGB_pred = modelxgb.predict(x_val)
    Y_RF_pred = modelRF.predict(x_val)
    Y_LR_pred = modelLR.predict(x_val)

    # 计算 y_pred_sum 和 y_sum
    y_NN_sum = np.sum(Y_NN_pred)
    y_XGB_sum = np.sum(Y_XGB_pred)
    y_RF_sum = np.sum(Y_RF_pred)
    y_LR_sum = np.sum(Y_LR_pred)

    y_sum = np.sum(y_val)

    mse_NN_train = mean_squared_error(y_val, Y_NN_pred)
    mae_NN_train = mean_absolute_error(y_val, Y_NN_pred)
    r2_NN_train = r2_score(y_val, Y_NN_pred)
    pre_NN_train = (y_NN_sum - y_sum) / y_sum

    mse_XGB_train = mean_squared_error(y_val, Y_XGB_pred)
    mae_XGB_train = mean_absolute_error(y_val, Y_XGB_pred)
    r2_XGB_train = r2_score(y_val, Y_XGB_pred)
    pre_XGB_train = (y_XGB_sum - y_sum) / y_sum

    mse_RF_train = mean_squared_error(y_val, Y_RF_pred)
    mae_RF_train = mean_absolute_error(y_val, Y_RF_pred)
    r2_RF_train = r2_score(y_val, Y_RF_pred)
    pre_RF_train = (y_RF_sum - y_sum) / y_sum

    mse_LR_train = mean_squared_error(y_val, Y_LR_pred)
    mae_lR_train = mean_absolute_error(y_val, Y_LR_pred)
    r2_LR_train = r2_score(y_val, Y_LR_pred)
    pre_LR_train = (y_LR_sum - y_sum) / y_sum

    # test集合上的性能结果
    Y_NN_test = model.predict(X_test)
    Y_XGB_test = modelxgb.predict(X_test)
    Y_RF_test = modelRF.predict(X_test)
    Y_LR_test = modelLR.predict(X_test)

    # 计算 y_pred_sum 和 y_sum
    y_NN_sum_test = np.sum(Y_NN_test)
    y_XGB_sum_test = np.sum(Y_XGB_test)
    y_RF_sum_test = np.sum(Y_RF_test)
    y_LR_sum_test = np.sum(Y_LR_test)

    y_sum_test = np.sum(y_test)

    mse_NN_test = mean_squared_error(y_test, Y_NN_test)
    mae_NN_test = mean_absolute_error(y_test, Y_NN_test)
    r2_NN_test = r2_score(y_test, Y_NN_test)
    pre_NN_test = (y_NN_sum_test - y_sum_test) / y_sum_test

    mse_XGB_test = mean_squared_error(y_test, Y_XGB_test)
    mae_XGB_test = mean_absolute_error(y_test, Y_XGB_test)
    r2_XGB_test = r2_score(y_test, Y_XGB_test)
    pre_XGB_test = (y_XGB_sum_test - y_sum_test) / y_sum_test

    mse_RF_test = mean_squared_error(y_test, Y_RF_test)
    mae_RF_test = mean_absolute_error(y_test, Y_RF_test)
    r2_RF_test = r2_score(y_test, Y_RF_test)
    pre_RF_test = (y_RF_sum_test - y_sum_test) / y_sum_test

    mse_LR_test = mean_squared_error(y_test, Y_LR_test)
    mae_lR_test = mean_absolute_error(y_test, Y_LR_test)
    r2_LR_test = r2_score(y_test, Y_LR_test)
    pre_LR_test = (y_LR_sum_test - y_sum_test) / y_sum_test

    # 储存结果
    results1.loc[fold - 1] = [fold, mse_NN_train, mae_NN_train, r2_NN_train, mse_XGB_train, mae_XGB_train, r2_XGB_train,
                           mse_RF_train, mae_RF_train, r2_RF_train, mse_LR_train, mae_lR_train, r2_LR_train,
                              mse_NN_test, mae_NN_test, r2_NN_test, mse_XGB_test, mae_XGB_test, r2_XGB_test,
                              mse_RF_test, mae_RF_test, r2_RF_test, mse_LR_test, mae_lR_test, r2_LR_test
                              ]
    results2.loc[fold - 1] = [fold, pre_NN_train, pre_XGB_train, pre_RF_train, pre_LR_train,
                              pre_NN_test, pre_XGB_test, pre_RF_test, pre_LR_test,]
    '''
    modelxgb = xgb.XGBRegressor()
    modelRF = RandomForestRegressor()
    modelxgb.fit(x_train, y_train)
    y_pred = modelxgb.predict(x_val)
    mse_val = mean_squared_error(y_val, y_pred)
    print("xgboost loss"+str(mse_val))

#    val_results = pd.DataFrame({'True Values': y_val, 'Predicted Values': y_pred.flatten()})
 #   print(val_results)

    modelRF.fit(x_train, y_train)
    y_pred_rf = modelRF.predict(x_val)
    mse_va_rf = mean_squared_error(y_val, y_pred_rf)
    print("RF loss"+str(mse_va_rf))

    y_pred_test = modelxgb.predict(X_test)
    mse_test = mean_squared_error(y_pred_test, y_test)
    print("xgboost loss test"+str(mse_test))
    xgb_test_results = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test.flatten()})
    print("xgb"+str(xgb_test_results))
    '''


    fold_losses.append(history.history['loss'])

    # 在独立测试集上评估模型并保存结果
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    predicted_output = model.predict(X_test)
    test_results = pd.DataFrame({'True Values': y_test, 'Predicted Values': predicted_output.flatten()})
    print(test_results)
    print("NN"+str(test_loss))

    os.path.exists(file_path + '\my_model_weights_%d.h5' % fold)
    model.save_weights(file_path + '\my_model_weights_%d.h5' % fold)

    '''对每次的结果进行绘画
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
###看到了吗 恩 右键 然后 运行你自己来
    # 绘制 X1, X2 与 y 的关系
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='b', marker='o')

    # 绘制 X1, X2 与 y_predict 的关系
    ax.scatter(X_test[:, 0], X_test[:, 1], predicted_output, c='r', marker='^')

    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c='g', marker='*')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y / y_predict')
    '''

    # 绘制效果图
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    # 绘制 X1, X2 与 y 的关系
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='b', marker='o')

    # 绘制 X1, X2 与 y_predict 的关系
    ax.scatter(X_test[:, 0], X_test[:, 1], predicted_output, c='r', marker='^')

    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='g', marker='*')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y / y_predict')

    # 绘制输入参数与输出的关系
    fig2 = plt.figure(figsize=(15, 5))
    # 添加第一个子图
    ax1 = fig2.add_subplot(131, projection='3d')  # 这里的131表示1行3列的第1个
    ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, c='b', marker='o')
    ax1.scatter(X_test[:, 0], X_test[:, 1], predicted_output, c='r', marker='*')
    ax1.set_xlabel('V')
    ax1.set_ylabel('R')
    ax1.set_zlabel('I')
    ax1.set_title('First Subplot')

    # 添加第二个子图
    ax2 = fig2.add_subplot(132, projection='3d')  # 这里的132表示1行3列的第2个
    ax2.scatter(X_test[:, 0], X_test[:, 2], y_test, c='b', marker='o')
    ax2.scatter(X_test[:, 0], X_test[:, 2], predicted_output, c='r', marker='*')
    ax2.set_xlabel('V')
    ax2.set_ylabel('D')
    ax2.set_zlabel('I')
    ax2.set_title('Second Subplot')

    # 添加第三个子图
    ax3 = fig2.add_subplot(133, projection='3d')  # 这里的133表示1行3列的第3个
    ax3.scatter(X_test[:, 1], X_test[:, 2], y_test, c='b', marker='o')
    ax3.scatter(X_test[:, 1], X_test[:, 2], predicted_output, c='r', marker='*')
    ax3.set_xlabel('R')
    ax3.set_ylabel('D')
    ax3.set_zlabel('I')
    ax3.set_title('Third Subplot')

    plt.savefig('result\path_to_your_file.png', dpi=300)
    plt.show()

results1.to_csv('result\MSE_MAE_R2.csv',)
results2.to_csv('result\Precentage.csv',)

f.to_csv('final.csv')

# 绘制loss折线图
plt.figure(figsize=(10, 6))
for fold_loss in fold_losses:
    plt.plot(fold_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Each Fold')
plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.savefig('result\loss.png', dpi=300)
plt.show()

'''
# 将所有模型的测试结果保存为'final.xls'
final_results = pd.DataFrame()
final_results.to_excel('final.xls', index=False)
for fold in range(5):
    fold_result = pd.read_excel(f'fold_{fold + 1}_test_results.xlsx')
    final_results[f'Fold {fold + 1}'] = fold_result['Predicted Values']
'''




