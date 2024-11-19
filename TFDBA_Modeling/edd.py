import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# 生成示例散点数据
np.random.seed(123)
x = np.random.rand(200)
y = np.random.rand(200)
x1 = np.random.rand(200)
y1 = np.random.rand(200)
z = 6*x + -8*y + -80*(x**2) + 40*x*y + -80*(y**2) + np.random.randn(200) * 0.001  # 添加噪声
z2 = 6*x1 + -8*y1 + -50*(x1**2) + 30*x1*y1 + -80*(y1**2) + np.random.randn(200) * 0.001 + 90
# 绘制散点图
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')


# 使用 curve_fit 拟合曲面
def func(xy, a, b, c, d, e,f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y

popt, pcov = curve_fit(func, (x, y), z)
popt1, pcov1 = curve_fit(func, (x1, y1), z2)

# 绘制拟合的曲面图
xq, yq = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
z_fit = func((xq, yq), *popt)
ax.plot_surface(xq, yq, z_fit, color='r', alpha=0.5)

xq1, yq1 = np.meshgrid(np.linspace(min(x1), max(x1), 100), np.linspace(min(y1), max(y1), 100))
z_fit1 = func((xq1, yq1), *popt1)
ax.plot_surface(xq1, yq1, z_fit1, color='orange', alpha=0.5)

# 绘制散点图
ax.scatter(x, y, z, c='r', marker='o',s=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0,max(x))
ax.set_ylim(0,max(y))
ax.set_title('Fitted Surface')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
#ax.legend(['Fitted Surface', 'Scatter Plot'])
ax.view_init(elev=5,azim=120)
plt.savefig('4399_无网格.png')
plt.show()
