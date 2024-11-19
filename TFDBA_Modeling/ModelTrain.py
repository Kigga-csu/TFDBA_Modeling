from load_model import model_load
from load_model import tf_model_load
import pickle
import pandas as pd
import numpy as np#把所有标红的都下载 用pip install 记得去网上搜索版本对应 错误的版本会导致无法运行
import Check_Table
import matplotlib.pyplot as plt
import time

def objective_function(D_update, R_rated, V_rated, choice):
    if choice == True:
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model = model_load(fold=0)
    else:
        scaler = pickle.load(open('scaler2.pkl', 'rb'))
        model = tf_model_load()
    c = np.expand_dims(np.array([V_rated,R_rated,D_update]), axis=0)####这里需要根据表格具体的输入进行修改按照D，V，R的对应关系
    c = scaler.transform(c)
    out = model(c)
    return out  # 请根据实际情况修改这个函�?

def fitness_function(y_target, y_actual):
    # 适应度函数，衡量目标值与实际值之间的差异
    return abs(y_target - y_actual)

def particle_swarm_optimization_plot(I_objected, R_rated, V_rated,file_path,choice, swarm_size=50, max_iterations=40, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):

    particles = np.random.rand(swarm_size)
    velocities = np.random.rand(swarm_size)
    personal_best_positions = particles.copy()
    personal_best_fitness = np.zeros(swarm_size) + np.inf
    global_best_position = None
    global_best_fitness = np.inf

    # 新增：用于保存每次迭代的适应度值
    fitness_values = []
    min_fitness_values = []
    max_fitness_values = []

    for iteration in range(max_iterations):
        # 计算适应度并更新个体和全局最优
        for i in range(swarm_size):
            D_update = particles[i]
            y_actual = objective_function(D_update, R_rated, V_rated, choice)
            fitness = fitness_function(I_objected, y_actual)

            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i]

        for i in range(swarm_size):
            if personal_best_fitness[i] < global_best_fitness:
                global_best_fitness = personal_best_fitness[i]
                global_best_position = personal_best_positions[i]

        # 新增：保存这次迭代的平均适应度值，最大适应度值和最小适应度值
        fitness_values.append(np.mean(personal_best_fitness))
        min_fitness_values.append(np.min(personal_best_fitness))
        max_fitness_values.append(np.max(personal_best_fitness))

        # 更新粒子位置和速度
        for i in range(swarm_size):
            velocities[i] = inertia_weight * velocities[i] + \
                             cognitive_weight * np.random.rand() * (personal_best_positions[i] - particles[i]) + \
                             social_weight * np.random.rand() * (global_best_position - particles[i])
            particles[i] = particles[i] + velocities[i]
            particles[i] = max(0, min(1, particles[i]))
    '''
      # 新增：绘制适应度值的折线图
    plt.plot(fitness_values, 'y-', linewidth=2, label='Average Fitness')
    plt.plot(min_fitness_values, 'y-', label='Min Fitness')
    plt.plot(max_fitness_values, 'y-', label='Max Fitness')
    plt.fill_between(range(max_iterations), min_fitness_values, max_fitness_values, color='y', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
     '''
    return global_best_position,fitness_values,max_fitness_values,min_fitness_values

def particle_swarm_optimization_table(I_objected, R_rated, V_rated, file_path, choice, swarm_size=50, max_iterations=40, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
    # 初始化粒子群
    d3_min, d3_max = Check_Table.check_table(V_rated, R_rated, I_objected, file_path)
    particles = np.random.uniform(d3_min, d3_max, swarm_size)#这里改换成查表显示，不让粒子群随机抽�?
    velocities = np.random.rand(swarm_size)
    personal_best_positions = particles.copy()
    personal_best_fitness = np.zeros(swarm_size) + np.inf #数组
    global_best_position = None
    global_best_fitness = np.inf #标量变量
    # 新增：用于保存每次迭代的适应度值
    fitness_values = []
    min_fitness_values = []
    max_fitness_values = []
    start_time = time.time()

    for iteration in range(max_iterations):
        # 新增初始化个体最佳适应度和位置的索引
        best_particle_index = None
        #best_global_index = None

        # 对每个粒子进行迭代
        for i in range(swarm_size):
            D_update = particles[i]

            # 计算目标函数值
            y_actual = objective_function(D_update, R_rated, V_rated, choice)

            # 计算适应度
            fitness = fitness_function(I_objected, y_actual)

            # 更新个体最佳位置和适应度
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i]

            # 记录个体最佳适应度的索引
                if best_particle_index is None or personal_best_fitness[i] < personal_best_fitness[best_particle_index]:
                    best_particle_index = i

        # 更新全局最佳适应度和全局最佳位置
        if best_particle_index is not None and (
                global_best_position is None or personal_best_fitness[best_particle_index] < global_best_fitness):
            global_best_fitness = personal_best_fitness[best_particle_index]
            global_best_position = personal_best_positions[best_particle_index]


        # 保存每次迭代的平均适应度值，最大适应度值和最小适应度值
        fitness_values.append(np.mean(personal_best_fitness))
        min_fitness_values.append(np.min(personal_best_fitness))
        max_fitness_values.append(np.max(personal_best_fitness))

        # 更新粒子位置和速度
        velocities = inertia_weight * velocities + \
                     cognitive_weight * np.random.rand(swarm_size) * (personal_best_positions - particles) + \
                     social_weight * np.random.rand(swarm_size) * (global_best_position - particles)
        particles = particles + velocities
        particles = np.clip(particles, d3_min, d3_max)
        # 新增：绘制适应度值的折线图
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code executed in {execution_time} seconds.")
    '''
    plt.plot(fitness_values, 'r-', linewidth=2, label='Average Fitness')
    plt.plot(min_fitness_values, 'r-', label='Min Fitness')
    plt.plot(max_fitness_values, 'r-', label='Max Fitness')
    plt.fill_between(range(max_iterations), min_fitness_values, max_fitness_values, color='mistyrose', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
    '''
    return global_best_position,fitness_values,max_fitness_values,min_fitness_values

#c1 = particle_swarm_optimization_table(10.0,0.6,40.0,'data\data_fake.xls',True)
#c2 = particle_swarm_optimization_plot (10.0,0.6,40.0,'data\data_fake.xls',True)

global_best_position,fitness_values,max_fitness_values,min_fitness_values = particle_swarm_optimization_plot (10.0,2,40.0,'data\data_fake.xls',True)
global_best_position_t,fitness_values_t,max_fitness_values_t,min_fitness_values_t = particle_swarm_optimization_table (10.0,2,40.0,'data\data_fake.xls',True)


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(10, 6))
plt.plot(fitness_values, 'y-', linewidth=2, label='PSO')
plt.plot(fitness_values_t, 'r-', linewidth=2, label='IPSO')
plt.plot(min_fitness_values, color='khaki', linestyle='dashed')
plt.plot(max_fitness_values, color='khaki', linestyle='dashed')
plt.plot(min_fitness_values_t, color='lightcoral', linestyle='dashdot')
plt.plot(max_fitness_values_t, color='lightcoral', linestyle='dashdot')
plt.fill_between(range(40), min_fitness_values, max_fitness_values, color='khaki', alpha=0.6)
plt.fill_between(range(40), min_fitness_values_t, max_fitness_values_t, color='lightcoral', alpha=0.5)
#设置图框线粗细
bwith = 1.2 #边框宽度设置为2
TK = plt.gca()#获取当前坐标轴
TK.spines['bottom'].set_linewidth(bwith)#图框下边
TK.spines['left'].set_linewidth(bwith)#图框左边
TK.spines['top'].set_linewidth(bwith)#图框上边
TK.spines['right'].set_linewidth(bwith)#图框右边
plt.xlabel('Iterations',fontsize=16)
plt.ylabel('Fitness',fontsize=16)
plt.ylim(0, 10)
plt.xlim(0, 20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(np.arange(0, 22, 4))
plt.tick_params(axis='both',width=1,length=5,direction='in')
plt.legend(frameon=False,loc="upper right",fontsize='small')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.savefig('result\pso.png', dpi=300)
plt.show()



plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(10, 6))
plt.plot(fitness_values, 'y-', linewidth=2)
plt.plot(fitness_values_t, 'r-', linewidth=2)
#设置图框线粗细
bwith = 1.2 #边框宽度设置为2
TK = plt.gca()#获取边框
TK.spines['bottom'].set_linewidth(bwith)#图框下边
TK.spines['left'].set_linewidth(bwith)#图框左边
TK.spines['top'].set_linewidth(bwith)#图框上边
TK.spines['right'].set_linewidth(bwith)#图框右边
plt.ylim(0, 0.3)
plt.xlim(19,20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(np.arange(19, 20.2, 0.2))
plt.tick_params(axis='both',width=1,length=5,direction='in')
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.savefig('result\pso1.png', dpi=300)
plt.show()

#print(c)


