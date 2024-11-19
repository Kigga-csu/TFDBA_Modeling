import pandas as pd
import numpy as np

def find_closest_two(V, x):
    """
    在列表V中找到与x欧式距离最近的两个数。

    参数:
    V (list): 输入的列表。
    x (float): 输入的参数。

    返回:
    (float, float): 与x欧式距离最近的两个数。
    """
    # 计算列表中每个元素与x的欧式距离
    distances = [abs(v - x) for v in V]

    # 找出最小的两个距离的索引
    closest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:2]

    # 返回与x欧式距离最近的两个数
    return V[closest_indices[0]], V[closest_indices[1]]

def find_d3_range(d1, d2, d4 ,table_path):
    """
    根据给定的D1、D2和D4的值，找到满足这些条件的D3值的范围。

    参数:
    df (pd.DataFrame): 包含D1、D2、D3和D4列的数据框。
    d1 (float): D1的值。
    d2 (float): D2的值。
    d4 (float): D4的值。

    返回:
    pd.Series: 满足条件的D3值的范围。
    """
    df = pd.read_excel(table_path)
    conditions = (df['V'] > d1 - 2) & (df['V'] < d1 + 2) & (df['R'] > d2 - 0.8) & (df['R'] < d2 + 0.8) & (df['I'] > d4 - 1) & (df['I'] < d4 + 1)
    d3_range = df.loc[conditions, 'D']
    return d3_range



def find2_d3_range(df, V, R, I):
    """
    根据给定的V、R和I的值，找到满足这些条件的D3值的最大值和最小值。

    参数:
    df (pd.DataFrame): 包含D1、D2、D3和D4列的数据框。
    V (float): V的值。
    R (float): R的值。
    I (float): I的值。

    返回:
    (float, float): 满足条件的D3值的最大值和最小值。
    """
    V_unique_sorted = sorted(df['V'].unique())
    R_unique_sorted = sorted(df['R'].unique())

    # 条件1：确定D1列中与V绝对值距离最近的两个数字
    condition_1 = find_closest_two(V_unique_sorted, V)

    # 条件2：确定D2中与R最近的两个数字
    condition_2 = find_closest_two(R_unique_sorted, R)

    # 取出表格中同时满足以上三种条件的所有D3列数值
    D_values = df[(df['V'].isin(condition_1)) & (df['R'].isin(condition_2)) & (df['I'] >= I - 0.5) & (df['I'] <= I + 0.5)]['D']

    # 比较得到这部分数值的最大最小值
    d3_min = D_values.min()
    d3_max = D_values.max()

    return d3_min, d3_max

# 使用示例：
# 假设df是您的数据框

df = pd.read_excel('data_real.xls')


V = 51.2
R = 14.6
I = 14.5

V_unique_sorted = sorted(df['V'].unique())
R_unique_sorted = sorted(df['R'].unique())

condition_1 = find_closest_two(V_unique_sorted, V)
vvvv = find2_d3_range(df, V, R, I)


print(vvvv)