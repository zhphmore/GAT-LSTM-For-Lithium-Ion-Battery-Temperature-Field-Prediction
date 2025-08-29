# # # ************************************************************
# # case93_raw
# # 数据集：data_batterypack/data_20240802/case93_raw
# # 北大网盘下载链接：https://disk.pku.edu.cn/link/AA9BCF06CEBB9A4546B0FA8897632A9D3C
# # ********基本设置********
# # 算例数量
# num_file = 93
# # 算例的标号是否从0开始（True认为从0开始，False认为从1开始；即第一个文件是input_0.csv，还是input_1.csv？）
# file_id_start_0 = True
# # 时间步（等于output的csv的行数）（20个时间步，每个时间步长1min，总时间20min）
# num_timestep = 20
# # 节点数（等于output的csv的列数）
# num_node = 742826
# # 初始温度（默认293.15K）
# temperature_init = 293.15
# # ********模型设置********
# # 图的每个节点连了多少条边
# num_edge_neighbor = 5
# # 图的边数（例：一共742826个节点，每个节点连5条边，那么总共3714130条边）
# num_edge = num_node * num_edge_neighbor
# # lstm隐藏层的维度（可以灵活修改）
# lstm_hidden_size = 32
# # 训练多少个epoch
# num_epochs = 10
# # ****单时间步预测设置********
# # 拆成多少个小csv（例：93个csv算例，每个算例20个时间步，拆成93*20=1860个小csv）
# # 这些小csv存放在data_nn文件夹下，从0开始编号（例：从data_nn_00.csv到data_nn_1859.csv）
# num_nn = num_file * num_timestep
# # 这些小csv中，多少个作为训练集（例：1860个小csv，1800个作为训练集，60个作为测试集）
# len_train = 1800
# # ********路径设置********
# # 数据集存放的路径（训练好的网络也会存放在这个路径下）
# save_path = 'D:/projects/battery_case/battery_4/data/case93_raw/'
# # save_path = '/mnt/data/puhan/csvdata/case93_raw/'  # linux
# # 节点坐标文件的名字
# cor_file_name = 'nodes_coordinates.csv'
# # 节点坐标文件的行数是否是num_node（True认为是num_node×3，False认为是3×num_node）
# # （例：nodes_coordinates.csv是742826×3，行数是742826，所以设置为True）
# cor_node_shape_0 = True
# # ************************************************************


# ************************************************************
# case93_selected1000
# 数据集：data_batterypack/data_20240802/case93_selected1000
# 北大网盘下载链接：https://disk.pku.edu.cn/link/AA9BCF06CEBB9A4546B0FA8897632A9D3C
# ********基本设置********
# 算例数量
num_file = 93
# 算例的标号是否从0开始（True认为从0开始，False认为从1开始；即第一个文件是input_0.csv，还是input_1.csv？）
file_id_start_0 = True
# 时间步（等于output的csv的行数）（20个时间步，每个时间步长1min，总时间20min）
num_timestep = 20
# 节点数（等于output的csv的列数）
num_node = 1000
# 初始温度（默认293.15K）
temperature_init = 293.15
# ********模型设置********
# 图的每个节点连了多少条边
num_edge_neighbor = 6
# 图的边数（例：一共1000个节点，每个节点连6条边，那么总共6000条边）
num_edge = num_node * num_edge_neighbor
# lstm隐藏层的维度（可以灵活修改）
lstm_hidden_size = 32
# 训练多少个epoch
num_epochs = 10
# ****单时间步预测设置********
# 拆成多少个小csv（例：93个csv算例，每个算例20个时间步，拆成93*20=1860个小csv）
# 这些小csv存放在data_nn文件夹下，从0开始编号（例：从data_nn_00.csv到data_nn_1859.csv）
num_nn = num_file * num_timestep
# 这些小csv中，多少个作为训练集（例：1860个小csv，1800个作为训练集，60个作为测试集）
len_train = 1800
# ********路径设置********
# 数据集存放的路径（训练好的网络也会存放在这个路径下）
# save_path = 'D:/projects/battery_case/battery_4/data/case93_selected1000/'
save_path = '/mnt/data/puhan/csvdata/case93_selected1000/'  # linux
# 节点坐标文件的名字
cor_file_name = 'kmeans_1000points_place.csv'
# 节点坐标文件的行数是否是num_node（True认为是num_node×3，False认为是3×num_node）
# （例：kmeans_1000points_place.csv是3×1000，行数是3，所以设置为False）
cor_node_shape_0 = False
# ************************************************************


# # ************************************************************
# # selected_1234
# # 数据集：data_batterypack/data_20240517/group_data/point
# # 北大网盘下载链接：https://disk.pku.edu.cn/link/AA9BCF06CEBB9A4546B0FA8897632A9D3C
# # ********基本设置********
# num_file = 228
# file_id_start_0 = False
# num_timestep = 20
# num_node = 1234
# temperature_init = 293.15
# # ********模型设置********
# num_edge_neighbor = 6
# num_edge = num_node * num_edge_neighbor
# lstm_hidden_size = 16
# num_epochs = 10
# # ****单时间步预测设置********
# num_nn = num_file * num_timestep
# len_train = 4000
# # ********路径设置********
# save_path = 'D:/projects/battery_case/battery_4/data/selected1234/'
# # save_path = '/mnt/data/puhan/csvdata/case93_raw/selected1234/'  # linux
# cor_file_name = 'cor_points.csv'
# cor_node_shape_0 = True
# # ************************************************************
