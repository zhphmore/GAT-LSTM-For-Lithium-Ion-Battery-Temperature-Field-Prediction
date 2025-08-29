import pandas as pd
import torch
from torch_geometric.nn.pool import knn_graph

import GL_set

# ****************************************
# graph_edge_index.py
# 该程序读取节点坐标
# 根据坐标构造一个无向图
# 输出图的邻接矩阵的COO格式
# ****************************************

num_edge_neighbor = GL_set.num_edge_neighbor
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

# **********将点云转化为图**********
# 采用最邻近方法knn构造无向图
# 函数knn_graph的返回值是2×num_edge的矩阵，edge_index_knn是torch格式
edge_index_knn = knn_graph(data_cor_torch, k=num_edge_neighbor)
print("边连接矩阵（COO格式）： shape = ", edge_index_knn.shape)
# 保存COO格式的邻接矩阵
# 保存路径
edge_index_file = save_path + 'edge_index.csv'
# 先转化为DataFrame格式，再保存
df_edge_index = pd.DataFrame(data=edge_index_knn.numpy())
df_edge_index.to_csv(edge_index_file, encoding='utf-8', header=False, index=False)
print("edge index successfully saved")
print("edge index path: ", edge_index_file)
print("end")

# **********补充说明**********
# 对函数knn_graph的返回值的进一步说明：
# 返回值是COO格式的图的邻接矩阵，COO格式是2×num_edge的矩阵，num_edge是边数
# 什么是COO格式？
# 举例：
# 假设：第2号节点和第6号节点有边连接，第19号节点和第1号节点有边连接，第985号节点和第211号节点有边连接
# 因为是无向图，在这个例子下，一共6条边，所以是2×6的矩阵
# 那么：COO格式的图的邻接矩阵为：
# [[2, 6, 1, 19, 211, 985],
#  [6, 2, 19, 1, 985, 211]]
