import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

import GL_set
import GL_model

# ****************************************
# testing_2.py
# 该程序测试神经网络
# testing_2.py测试num_timestep个时间步的预测准确率，testing_1.py测试1个时间步的预测准确率
# 注意
# 程序运行前，记得修改模型参数保存的位置
# ****************************************

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序读取的模型的文件名称
name_pth = 'gl01.pth'

# **********读取设置**********
num_timestep = GL_set.num_timestep
num_node = GL_set.num_node
num_edge = GL_set.num_edge
lstm_hidden_size = GL_set.lstm_hidden_size
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

# 数据文件位置
data_process_file_name = save_path + 'data_nn/data_nn_'

# **********测试20个时间步的预测准确率**********

# ********************
# 挑一个算例进行测试
# 具体挑哪个，可以由你定，在此处修改
# test_file_id: 1 ~ num_file
test_file_id = 45
# ********************
# nn_id指的是挑出来的那个算例对应data_nn文件夹内的哪个csv
nn_id = (test_file_id - 1) * num_timestep
# 举例：case93_selected1000
# 如果test_file_id=19，表示用第19号算例进行测试，也就是用input_18.csv和output_18.csv进行测试
# 被拆成的小csv，对应data_nn文件夹内的从data_nn_360.csv到data_nn_379.csv，所以nn_id=360

# 由t0时刻，预测t1时刻
data_process_file = data_process_file_name + str(nn_id) + '.csv'
data_df = pd.read_csv(data_process_file, encoding='utf-8')
data_i = Data(x=torch.from_numpy(data_df['x'].to_numpy()),
              xTB=torch.from_numpy(data_df['xTB'].to_numpy()),
              cur=torch.from_numpy(data_df['cur'].to_numpy()[0:num_timestep]),
              y=torch.from_numpy(data_df['y'].to_numpy()))
out = loaded_model(data_i, edge_index_knn)
loss = criterion(out, data_i.y)

print("\nid ", 1, ":")
# print("real: ", data_i.y, "\npred: ", out)
print("real_max: ", max(data_i.y), "\npred_max: ", max(out))
print("real_min: ", min(data_i.y), "\npred_min: ", min(out))
print("loss: ", loss)

# 由t1时刻，依次预测t2, t3, ..., t_num_timestep时刻
for i in range(num_timestep - 1):
    data_process_file = data_process_file_name + str(nn_id + 1 + i) + '.csv'
    data_df = pd.read_csv(data_process_file, encoding='utf-8')
    data_i = Data(x=out,
                  xTB=torch.from_numpy(data_df['xTB'].to_numpy()),
                  cur=torch.from_numpy(data_df['cur'].to_numpy()[0:num_timestep]),
                  y=torch.from_numpy(data_df['y'].to_numpy()))
    out = loaded_model(data_i, edge_index_knn)
    loss = criterion(out, data_i.y)

    print("\nid ", i + 2, ":")
    # print("real: ", data_i.y, "\npred: ", out)
    print("real_max: ", max(data_i.y), "\npred_max: ", max(out))
    print("real_min: ", min(data_i.y), "\npred_min: ", min(out))
    print("loss: ", loss)
print("test model: ", save_path_gl)
print("end")
