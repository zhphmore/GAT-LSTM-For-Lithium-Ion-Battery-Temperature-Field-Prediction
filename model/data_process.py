import numpy as np
import pandas as pd
import torch

import GL_set

# ****************************************
# data_process.py
# 单时间步预测
# 因此要对输入输出做一些修改，相当于数据集预处理
# 预处理后的数据集存放在data_nn文件夹
# 之后，神经网络的训练和测试，都直接从data_nn文件夹读取数据
# （注意：
#   运行该程序前，请先在GL_set.save_path目录下新建一个名叫data_nn的空文件夹！
#   该程序读取input, output文件夹内的数据进行预处理，然后将预处理后的数据存放在新建的data_nn文件夹内
#   即：data_nn, input, output这三个文件夹都在同一个目录下，该目录是GL_set.save_path）
# ****************************************

# **********读取设置**********
num_file = GL_set.num_file
file_id_start_0 = GL_set.file_id_start_0
num_timestep = GL_set.num_timestep
temperature_init = GL_set.temperature_init
save_path = GL_set.save_path
cor_file_name = GL_set.cor_file_name
cor_node_shape_0 = GL_set.cor_node_shape_0

# **********读取坐标点云数据**********
# 读取节点坐标文件，存放到data_cor_torch
# 节点坐标文件路径
cor_file = save_path + cor_file_name
data_cor_numpy = pd.read_csv(cor_file, encoding='utf-8', header=None).to_numpy()
# 检查节点坐标文件是num_node×3，还是3×num_node
# GL_set.cor_node_shape_0为True，表示是num_node×3
# 如果是3×num_node，读取后要转置
if not cor_node_shape_0:
    data_cor_numpy = data_cor_numpy.T
# 将numpy格式转化为torch格式
data_cor_torch = torch.tensor(data_cor_numpy)
# 检查读取是否正确
print("点云坐标矩阵： shape = ", data_cor_torch.shape)
num_node = data_cor_torch.shape[0]
print("节点数量： num_node = ", num_node)

# **********数据集预处理**********

# 思想：
# 目标是训练一个做单时间步预测的神经网络：
# 神经网络的输入是前一个时刻的数据，输出是后一个时刻的数据
# input, output文件夹内，有num_file个csv，即num_file个算例
# 每个算例有num_timestep个时刻
# 所以，每个算例可以拆分成num_timestep个小算例
# 这样一共就有了num_file*num_timestep个小算例，即拆出了num_file*num_timestep个小csv
# 把拆出的小csv保存在data_nn文件夹内
# 如果还不理解，请看程序结尾的补充说明

# 程序会读取input, output文件夹内的数据进行预处理
# 然后将预处理后的数据存放在新建的data_nn文件夹内
# 程序运行前：data_nn是一个新建的空文件夹
# 程序运行后：data_nn文件夹内会生成num_file*num_timestep个小csv
input_file_name = save_path + 'input/input_'
output_file_name = save_path + 'output/output_'
save_path_data_process = save_path + 'data_nn/data_nn_'

# data_nn文件夹内新生成的每个csv，是num_node×4的
# 第一列是"x": 当前时刻各个节点的温度，维度为num_node
# 第二列是"xTB": 与各个节点的x坐标相同的边界上的边界温度，维度为num_node
# 第三列是"cur": 从初始时刻到当前时刻的电流，维度为num_timestep
# 但为了DataFrame的格式整齐，方便保存，把"cur"的维度也统一为num_node，多余的部分填0，也就是实际上只有前num_timestep有用
# 第四列是"y": 下一时刻各个节点的温度，维度为num_node
# 对于单时间步预测的神经网络，"x"、"xTB"、"cur"是神经网络的输入，"y"是神经网络的输出
data_TB = np.zeros(num_node)
cur_init = np.zeros(num_timestep)
data_cur = np.zeros(num_node)
x_init = np.zeros(num_node)
for i in range(num_node):
    x_init[i] = temperature_init

cnt = 0

# **********
# for循环：读取input和output文件夹内的csv
for fid in range(num_file):
    # 读取input和output文件夹内的csv
    # 检查算例的标号是否从0开始（即第一个文件是input_0.csv还是input_1.csv？）
    if not file_id_start_0:
        fid = fid + 1
    input_file = input_file_name + str(fid) + '.csv'
    output_file = output_file_name + str(fid) + '.csv'
    data_input = pd.read_csv(input_file, encoding='utf-8', header=None).to_numpy()
    data_output = pd.read_csv(output_file, encoding='utf-8', header=None).to_numpy()
    # 举例：case93_selected1000
    # data_input是20×4的矩阵，data_output是20×1000的矩阵

    # 温度边界条件
    # 边界温度指的是，与电池包外表面接触的空气或水的温度
    # 空气或水的温度沿空间x轴方向，呈二次函数方式变化
    # 三个温度边界条件，单位分别为：a0: K, a1: K/m, a2: K/m^2
    # xx的单位为米，空间x轴坐标为xx处的边界温度为：a2 * (xx ^ 2) + a1 * xx + a0
    # data_TB是维度为num_node的数组，意义为：与各个节点的x坐标相同的边界上的边界温度
    a0 = data_input[0][3]
    a1 = data_input[0][2]
    a2 = data_input[0][1]
    for i in range(num_node):
        xx = data_cor_numpy[i][0]
        data_TB[i] = a2 * xx * xx + a1 * xx + a0
    # print(data_TB.shape)
    # 举例：case93_selected1000
    # data_TB是维度为1000的数组，储存的是：与各个节点的x坐标相同的边界上的边界温度

    # 原始电流
    # cur_init是维度为1×num_timestep的数组，存放各时间步的电流，电流的单位为安培
    for i in range(num_timestep):
        cur_init[i] = data_input[i][0]
    # print(data_cur.shape)
    # 举例：case93_selected1000
    # cur_init存放20个时间步的电流

    # 根据时间步调整电流和温度
    # 第0个时间步
    for j in range(num_timestep):
        data_cur[j] = 0
    data_cur[num_timestep - 1] = cur_init[0]

    data_tmp = {"x": x_init, "xTB": data_TB, "cur": data_cur, "y": data_output[0]}
    # 先转化为DataFrame格式，再保存到data_nn文件夹
    df_tmp = pd.DataFrame(data_tmp)
    save_file_cnt = save_path_data_process + str(cnt) + '.csv'
    df_tmp.to_csv(save_file_cnt, encoding='utf-8', index=False)
    cnt += 1

    # 根据时间步调整电流和温度
    # for循环：第1到第19个时间步
    for i in range(num_timestep - 1):
        for j in range(num_timestep):
            data_cur[j] = 0
        data_cur[num_timestep - i - 2:num_timestep] = cur_init[0:i + 2]

        data_tmp = {"x": data_output[i], "xTB": data_TB, "cur": data_cur, "y": data_output[i + 1]}
        # 先转化为DataFrame格式，再保存到data_nn文件夹
        df_tmp = pd.DataFrame(data_tmp)
        save_file_cnt = save_path_data_process + str(cnt) + '.csv'
        df_tmp.to_csv(save_file_cnt, encoding='utf-8', index=False)
        cnt += 1

    #
    print("the number of csv generated: ", cnt)

# 共计生成的文件数
print("totally generated: ", cnt)
print("data generated path: ", save_path + 'data_nn/')
print("end")

# **********补充说明**********
# 举例：case93_selected1000
# input, output文件夹内，有93个算例，每个算例有20个时刻
# 因此程序运行后，data_nn文件夹内会生成93*20=1860个小csv
# 每个小csv里，存放的是1000×4的矩阵，也就是有4列
# 第一列是"x": 当前时刻1000个节点的温度
# 第二列是"xTB": 与1000个节点各点的x坐标相同的边界上的边界温度
# 第三列是"cur": 该列只有前20行有用（但为了方便保存，也扩充成1000行，后面添0）
# 第四列是"y": 下一时刻1000个节点的温度
# 小csv之后会用于单时间步预测的神经网络，"x"、"xTB"、"cur"是神经网络的输入，"y"是神经网络的输出
# 对"cur"的进一步说明：
# 例如：input_82.csv，电流为[136.0, 385.95, 587.47, 740.526, 845.21, 901.43, ..., -868.56, -779.48, -641.96]
# 由input_82.csv和output_82.csv会拆分出20个小csv：data_nn_1640.csv到data_nn_1659.csv
# 第0个时间步，data_nn_1640.csv，第三列的电流"cur"为[0, 0, 0, 0, ..., 0, 0, 0, 0, 136.0]
# 第1个时间步，data_nn_1641.csv，第三列的电流"cur"为[0, 0, 0, 0, ..., 0, 0, 0, 136.0, 385.95]
# 第2个时间步，data_nn_1642.csv，第三列的电流"cur"为[0, 0, 0, 0, ..., 0, 0, 136.0, 385.95, 587.47]
# 第19个时间步，data_nn_1659.csv，第三列的电流"cur"为[136.0, 385.95, 587.47, 740.526, ..., -868.56, -779.48, -641.96]
