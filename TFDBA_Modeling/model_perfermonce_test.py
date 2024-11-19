
import pandas as pd

import matplotlib.pyplot as plt

# 保存结果到xls文件


def real_simulink_data_scatt():
    data_fake = pd.read_excel('data/data_fake.xls')
    data_real = pd.read_excel('data/data_real.xls')
    # 提取输入和输出数据
    X_fake = data_fake.iloc[:, 0:3].values
    y_fake = data_fake.iloc[:, 3].values
    X_real = data_real.iloc[:, 0:3].values
    y_real = data_real.iloc[:, 3].values


    fig = plt.figure(figsize=(15, 5))
    # 添加第一个子图
    ax1 = fig.add_subplot(131,projection='3d')  # 这里的131表示1行3列的第1个
    ax1.scatter(X_fake[:, 0], X_fake[:, 1], y_fake, c='b', marker='o')
    ax1.scatter(X_real[:, 0], X_real[:, 1], y_real, c='r', marker='*')
    ax1.set_xlabel('V')
    ax1.set_ylabel('R')
    ax1.set_zlabel('I')
    ax1.set_title('First Subplot')

    # 添加第二个子图
    ax2 = fig.add_subplot(132, projection='3d')  # 这里的132表示1行3列的第2个
    ax2.scatter(X_fake[:, 0], X_fake[:, 2], y_fake, c='b', marker='o')
    ax2.scatter(X_real[:, 0], X_real[:, 2], y_real, c='r', marker='*')
    ax2.set_xlabel('V')
    ax2.set_ylabel('D')
    ax2.set_zlabel('I')
    ax2.set_title('Second Subplot')

    # 添加第三个子图
    ax3 = fig.add_subplot(133,projection='3d')  # 这里的133表示1行3列的第3个
    ax3.scatter(X_fake[:, 1], X_fake[:, 2], y_fake, c='b', marker='o')
    ax3.scatter(X_real[:, 1], X_real[:, 2], y_real, c='r', marker='*')
    ax3.set_xlabel('R')
    ax3.set_ylabel('D')
    ax3.set_zlabel('I')
    ax3.set_title('Third Subplot')

    plt.savefig('real_fake.png', dpi=900)
    plt.show()

real_simulink_data_scatt()