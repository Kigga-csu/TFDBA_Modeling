import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# 读取 Excel 文件
df = pd.read_excel('data/data_fake1.xls')
ff = pd.read_excel('data/data_real.xls')
# 标准化每列数据
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.random.rand(100, 1)
z = np.sin(x) + np.cos(y) + np.random.randn(100, 1) * 0.1
z2 = z + 0.2
# 随机抽取 2000 个样本
sample_df = df.sample(n=2000)
#sample_ff = ff_standardized.sample(n=2000)
# 提取 x、y、z 列
x1 = sample_df.iloc[:, 0].values
y1 = sample_df.iloc[:, 1].values
z1 = sample_df.iloc[:, 3].values

x2 = ff.iloc[:, 0].values
y2 = ff.iloc[:, 1].values
z2 = ff.iloc[:, 3].values

# 定义拟合函数
def func(xy, a, b, c, d, e, ):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2

# 使用 curve_fit 拟合曲面
popt, pcov = curve_fit(func, (x, y), z)
popt1, pcov1 = curve_fit(func, (x2, y2), z2)

# 绘制散点图和拟合的曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(x1, y1, z1, c='yellow', alpha=0.6)
ax.scatter(x2, y2, z2, c='red', alpha=0.6)
# 生成曲面网格
xq, yq = np.meshgrid(np.linspace(min(x1), max(x1), 100), np.linspace(min(y1), max(y1), 100))
z_fit = func((xq, yq), *popt)

xq1, yq1 = np.meshgrid(np.linspace(min(x2), max(x2), 100), np.linspace(min(y2), max(y2), 100))
z_fit1 = func((xq1, yq1), *popt1)

# 绘制拟合曲面
ax.plot_surface(xq, yq, z_fit, color='yellow', alpha=1)
ax.plot_surface(xq1, yq1, z_fit1, color='orange', alpha=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Scatter Plot with Fitted Surface')
ax.view_init(elev=10,azim=120)

plt.show()