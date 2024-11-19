import tensorflow as tf
from BP_NN import build_model
import os
from sklearn.model_selection import KFold,train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def load_DCDC_data (file='F:\zyj20240605\zyj\data\data_real.xls'):
    # 读取xlsx文件
    data = pd.read_excel(file)

    # 提取输入和输出数据
    X = data.iloc[:, 0:3].values
    y = data.iloc[:, 3].values
    # 数据分割
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open('scaler2.pkl', 'wb'))
    return X_train, X_test, Y_train, y_test

# 加载预训练的模型权重
def model_load(fold=0):
    model = build_model()
    file_path_1 = os.getcwd()
    model.load_weights(file_path_1+'\model_paramater\my_model_weights_%d.h5' % fold)
    return model

def tf_model_load():
    model = build_model()
    file_path_1 = os.getcwd()
    model.load_weights(file_path_1 + '\model_paramater\my_tf_model_weights.h5')
    return model

def finetune_final_layer(M=None,x_experiment=None,y_experiment=None):

    for layer in M.layers[:-2]:  # 这里"-2"表示最后两层，你可以根据需要调整
        layer.trainable = False

    # 确保最后几层是可训练的
    for layer in M.layers[-2:]:
        layer.trainable = True
    # 打印出模型M的摘要
    M.summary()

    # 然后，你可以像往常一样编译和训练你的模型
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    M.compile(optimizer=optimizer, loss='mean_absolute_error')
    history = M.fit(x_experiment, y_experiment,validation_split=0.2,epochs=2000, batch_size=32,)
    return M


def finetune_extend_layer(M=None,x_experiment=None,y_experiment=None):

    # 冻结原有模型的所有层
    for layer in M.layers:
        layer.trainable = False

    # 在模型后面添加两层全连接层
    M.add(tf.keras.layers.Dense(10, activation='relu'))
    M.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # 然后，你可以像往常一样编译和训练你的模型
    M.compile(optimizer='adam', loss='categorical_crossentropy')
    history2 = M.fit(x_experiment, y_experiment,validation_split=0.1)
    return M



#############compare experment############
def compare():
    #############data_load###############
    X_train, X_test, Y_train, y_test = load_DCDC_data()

    #############model finting##########
    Original_Model = model_load()
    tf_model = finetune_final_layer(Original_Model, X_train, Y_train)
    # 实验数据模型
    model = build_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    history1 = model.fit(X_train, Y_train, validation_split=0.2,
                         epochs=2000, batch_size=32, verbose=1)

    ##############model testing#############
    Original_loss_test = Original_Model.evaluate(X_test, y_test, verbose=0)
    Original_loss_train = Original_Model.evaluate(X_train, Y_train, verbose=0)
    test_loss_test = tf_model.evaluate(X_test, y_test, verbose=0)
    test_loss_train = tf_model.evaluate(X_train, Y_train, verbose=0)
    model_loss_test = model.evaluate(X_test, y_test, verbose=0)
    model_loss_train = model.evaluate(X_train, Y_train, verbose=0)

    ###测试结果结算#####
    tf_output = tf_model.predict(X_test)
    Original_output = Original_Model.predict(X_test)
    model_output = model.predict(X_test)

    y_sum = np.sum(y_test)
    Original_pred_sum = np.sum(Original_output)
    tf_sum = np.sum(tf_output)
    model_sum = np.sum(model_output)

    mse_TF_train = mean_squared_error(y_test, tf_output)
    mape_TF_train = np.mean(np.abs((tf_output - y_test) / y_test))
    r2_TF_train = r2_score(y_test, tf_output)
    pre_TF_train = (tf_sum - y_sum) / y_sum

    mse_Original_train = mean_squared_error(y_test, Original_output)
    mape_Original_train = np.mean(np.abs((Original_output - y_test) / y_test))
    r2_Original_train = r2_score(y_test, Original_output)
    pre_Original_train = (Original_pred_sum - y_sum) / y_sum

    mse_model_train = mean_squared_error(y_test, model_output)
    mape_model_train = np.mean(np.abs((model_output - y_test) / y_test))
    r2_model_train = r2_score(y_test, model_output)
    pre_model_train = (model_sum - y_sum) / y_sum

    results = pd.DataFrame(columns=['mse_TF_train', 'mape_TF_train', 'r2_TF_train', 'pre_TF_train',
                                    'mse_Original_train', 'mape_Original_train', 'r2_Original_train',
                                    'pre_Original_train',
                                    'mse_model_train', 'mape_model_train', 'r2_model_train', 'pre_model_train', ])
    results.loc[0] = [mse_TF_train, mape_TF_train, r2_TF_train, pre_TF_train,
                      mse_Original_train, mape_Original_train, r2_Original_train, pre_Original_train,
                      mse_model_train, mape_model_train, r2_model_train, pre_model_train
                      ]
    results.to_csv('compare_data_train.csv')

    test_results = pd.DataFrame({'True Values': y_test, 'Predicted Values': tf_output.flatten()})
    print(test_results)
    print("simulation_test" + str(Original_loss_test))
    print("simulation_train" + str(Original_loss_train))
    print("experiment_test" + str(model_loss_test))
    print("experiment_train" + str(model_loss_train))
    print("transfer_test" + str(test_loss_test))
    print("transfer_train" + str(test_loss_train))

    #############tf_model saving###############
    file_path = os.getcwd()
    #os.path.exists(file_path + '\my_tf_model_weights.h5')
    tf_model.save_weights(file_path + '\my_tf_model_weights.h5')

    '''
    tf_model_1 = finetune_extend_layer(X_train, Y_train)
    tf_model_2 = finetune_final_layer(X_train, Y_train)
    test_loss_1 = tf_model_1.evaluate(X_test, y_test, verbose=0)
    test_loss_2 = tf_model_2.evaluate(X_test, y_test, verbose=0)
    print("test_loss_1" + str(test_loss_1))
    print("test_loss_2"+str(test_loss_2))
    '''
if __name__ == '__main__':
    compare()