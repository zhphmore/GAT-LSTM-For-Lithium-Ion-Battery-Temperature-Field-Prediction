import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing


# ****************************************
# GL_model.py
# 定义神经网络的模型
# 将图神经网络GNN和长短期记忆模型LSTM结合了起来，因此取了GNN和LSTM的首字母命名为GL（自创的，随便取的）
# 思想是试图将GNN的空间预测能力和LSTM的时间预测能力相结合
# 该模型进行单时间步的预测
# 该模型效果并不好，只能作为一个参考，可以魔改，或者换用其它的方法
# ****************************************

# **********图神经网络部分**********
# 参考资料：
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
# https://zhuanlan.zhihu.com/p/382236236

class GNN1(MessagePassing):
    def __init__(self, num_edge):
        super().__init__(aggr='add', flow='source_to_target')
        self.num_edge = num_edge

        # self.att是可被训练参数，意义类似于传热系数
        self.att = Parameter(torch.empty(num_edge, 1))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # x是num_node×1的矩阵，torch格式（特别注意x并非num_node的数组，而是矩阵，也就是说x是二维的）
        # edge_index是2×num_edge的矩阵，torch格式
        # 函数propagate会调用函数message
        # 更多详细信息请学习：图神经网络的消息传递范式
        out = self.propagate(edge_index=edge_index, x=x)
        return out

    def message(self, x_i, x_j):
        # x_i, x_j都是num_edge×1的矩阵，torch格式
        # 注意区分：x是节点数×1的矩阵，然而x_i, x_j都是边数×1的矩阵，并且都是二维的，是矩阵不是数组
        # x_j代表着相邻节点的温度，x_i代表着中心节点的温度
        # 每一条边连接两个节点，x_j与x_i二者相减代表该边连接的两个节点的温度差
        # att意义类似于传热系数，温度差乘传热系数，类似于是相邻节点对中心节点的传热
        ans = (x_j - x_i) * self.att
        return ans


# **********GL**********
# GL是自己编的神经网络

class GL(torch.nn.Module):
    def __init__(self, num_node, num_edge, num_timestep, lstm_hidden_size=8):
        super(GL, self).__init__()
        self.num_node = num_node
        self.num_edge = num_edge
        self.num_timestep = num_timestep
        self.lstm_hidden_size = lstm_hidden_size

        self.paraTB = Parameter(torch.empty(num_node))
        nn.init.normal_(self.paraTB)
        self.x30 = torch.ones(num_node)

        self.gnn1 = GNN1(num_edge)
        self.lstm1 = nn.LSTM(1, lstm_hidden_size, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(num_timestep * lstm_hidden_size, 1)
        self.linear2 = nn.Linear(num_node, num_node)
        nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, data, edge_index):
        # x, xTB都是num_node的数组，cur是num_timestep的数组，都是一维的，都是torch格式
        # edge_index是2×num_edge的矩阵，torch格式
        x, xTB, cur = data.x, data.xTB, data.cur

        # 空间部分
        # 热传导，节点温度到节点温度
        # x1是num_node的数组，代表着热传导给予各节点的温度增量
        # GNN1所接受的x是num_node×1的矩阵，因此x.unsqueeze(-1)要先把数组升成矩阵
        x1 = self.gnn1.forward(x.unsqueeze(-1), edge_index)
        x1 = x1.squeeze(1)

        # 热对流，边界温度到节点温度
        # x2是num_node的数组，代表着热对流给予各节点的温度增量
        # x2 = (xTB - x) * self.paraTB
        x21 = self.linear2(xTB - x)
        x2 = self.linear3(x21)

        # 时间部分
        # 电流生热，电流到节点温度
        # x3是num_node的数组，代表着电流生热给予各节点的温度增量
        # LSTM所接受的cur是num_timestep×1的矩阵，因此cur.unsqueeze(-1)要先把数组升成矩阵
        cur = cur.to(torch.float32).unsqueeze(-1)
        # out3是LSTM输出的隐藏层参数，是num_timestep×lstm_hidden_size的矩阵
        # 用torch.flatten把out3从二维矩阵展平成一维的
        out3, _ = self.lstm1(cur)
        out3 = torch.flatten(out3)
        x3 = self.linear1(out3)
        x3 = x3 * self.x30

        # x1, x2, x3分别代表当前时刻的各节点温度、边界温度、电流，造成的各节点的温度增量
        # y代表下一时刻的各节点温度
        y = x + x1 + x2 + x3

        return y
