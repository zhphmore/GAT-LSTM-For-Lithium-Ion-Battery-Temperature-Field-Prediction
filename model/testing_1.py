import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

import GL_set
import GL_model

# ****************************************
# testing_1.py
# 该程序测试神经网络
# testing_1.py测试1个时间步的预测准确率，testing_2.py测试num_timestep个时间步的预测准确率
# 注意
# 程序运行前，记得修改模型参数保存的位置
# ****************************************

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序读取的模型的文件名称
name_pth = 'gl01.pth'

# **********读取设置**********
num_timestep = GL_set.num_timestep
# num_node = GL_set.num_node
# num_edge = GL_set.num_edge
# lstm_hidden_size = GL_set.lstm_hidden_size
len_train = GL_set.len_train
save_path = GL_set.save_path

# **********读取图结构**********
# 读取edge_index.csv（edge_index.csv是由graph_edge_index.py产生的）
edge_index_file = save_path + 'edge_index.csv'
edge_index_knn = pd.read_csv(edge_index_file, encoding='utf-8', header=None).to_numpy()
# edge_index_knn是2×num_edge的矩阵，torch格式
edge_index_knn = torch.tensor(edge_index_knn)
print("边连接矩阵（COO格式）： shape = ", edge_index_knn.shape)

# **********加载模型**********
# 注意
# 记得修改模型参数保存的位置
save_path_gl = save_path + name_pth
loaded_model = torch.load(save_path_gl)
# loaded_model = GL_model.GL(num_node, num_edge, num_timestep, lstm_hidden_size)
# 损失函数
# criterion = nn.MSELoss()
criterion = nn.L1Loss(reduction='mean')
print("test model: ", save_path_gl)
print("model loaded successfully")

# **********测试集大小及位置**********
data_process_file_name = save_path + 'data_nn/data_nn_'
shuffle_file = save_path + 'shuffle_id.csv'
shuffle_id = pd.read_csv(shuffle_file, encoding='utf-8', header=None).to_numpy()
# 抛开训练集，剩下的部分就是测试集
len_test = shuffle_id.shape[0] - len_train

# **********测试1个时间步的预测准确率**********
loaded_model.eval()
total_loss = 0
# for循环：测试集的大小
for i in range(len_test):
    # 读取测试集的文件
    data_process_file = data_process_file_name + str(shuffle_id[len_train + i][0]) + '.csv'
    data_df = pd.read_csv(data_process_file, encoding='utf-8')
    # 将DataFrame格式转化为torch格式
    data_i = Data(x=torch.from_numpy(data_df['x'].to_numpy()),
                  xTB=torch.from_numpy(data_df['xTB'].to_numpy()),
                  cur=torch.from_numpy(data_df['cur'].to_numpy()[0:num_timestep]),
                  y=torch.from_numpy(data_df['y'].to_numpy()))

    # 模型输出
    out = loaded_model(data_i, edge_index_knn)
    loss = criterion(out, data_i.y)
    print("id ", i, ":")
    # print("real: ", data_i.y, "\npred: ", out)
    print("real_max: ", max(data_i.y), "\npred_max: ", max(out))
    print("real_min: ", min(data_i.y), "\npred_min: ", min(out))
    print("loss: ", loss)
    total_loss += loss
# 平均损失
avg_loss = total_loss / (len_test - 1)
print("\ntest length: ", len_test, "\naverage loss: ", avg_loss)
print("test model: ", save_path_gl)
print("end")
