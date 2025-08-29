import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import *

import GL_set
import GL_model

# ****************************************
# training_v2.py
# 该程序训练神经网络
# training_v2.py与training.py的不同之处：
# training.py是先读取全部的数据到内存中，再进行训练
# training_v2.py是一边训练一边读取数据的，更节省内存
# 注意
# 程序运行前，记得修改模型参数保存的位置
# ****************************************

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序输出的训练好的模型的文件名称
name_pth = 'gl02.pth'

# **********读取设置**********
num_timestep = GL_set.num_timestep
num_node = GL_set.num_node
num_edge = GL_set.num_edge
lstm_hidden_size = GL_set.lstm_hidden_size
num_epochs = GL_set.num_epochs
len_train = GL_set.len_train
save_path = GL_set.save_path

# **********读取图结构**********
# 读取edge_index.csv（edge_index.csv是由graph_edge_index.py产生的）
edge_index_file = save_path + 'edge_index.csv'
edge_index_knn = pd.read_csv(edge_index_file, encoding='utf-8', header=None).to_numpy()
# edge_index_knn是2×num_edge的矩阵，torch格式
edge_index_knn = torch.tensor(edge_index_knn)
print("边连接矩阵（COO格式）： shape = ", edge_index_knn.shape)

# **********数据的相关信息**********
# 获取打乱后的训练集顺序
# 读取shuffle_id.csv（shuffle_id.csv是由data_id_shuffle.py产生的）
shuffle_file = save_path + 'shuffle_id.csv'
shuffle_id = pd.read_csv(shuffle_file, encoding='utf-8', header=None).to_numpy()
# 预处理后的数据是放在data_nn文件夹里的（data_nn文件夹里的csv是由data_process.py产生的）
data_process_file_name = save_path + 'data_nn/data_nn_'

# **********模型训练设置**********
# 设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型
model = GL_model.GL(num_node, num_edge, num_timestep, lstm_hidden_size)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# 损失函数
# criterion = nn.MSELoss()
criterion = nn.L1Loss(reduction='mean')

# **********训练模型**********
# for循环：训练迭代epoch轮
for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    for i in range(len_train):
        # 读取data_nn文件夹里的csv
        data_process_file = data_process_file_name + str(shuffle_id[i][0]) + '.csv'
        data_df = pd.read_csv(data_process_file, encoding='utf-8')
        # 将DataFrame格式转化为torch格式
        data_i = Data(x=torch.from_numpy(data_df['x'].to_numpy()),
                      xTB=torch.from_numpy(data_df['xTB'].to_numpy()),
                      cur=torch.from_numpy(data_df['cur'].to_numpy()[0:num_timestep]),
                      y=torch.from_numpy(data_df['y'].to_numpy()))

        optimizer.zero_grad()
        out = model(data_i, edge_index_knn)
        loss = criterion(out, data_i.y)
        total_loss += loss
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len_train
    print("\nepoch ", epoch + 1, " : ", avg_loss)

# **********保存模型**********
# 注意
# 记得修改模型参数保存的位置
save_path_gl = save_path + name_pth
torch.save(model, save_path_gl)
print("trained model successfully saved")
print("model save path: ", save_path_gl)
print("end")