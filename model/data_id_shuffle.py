import numpy as np
import pandas as pd

import GL_set

# ****************************************
# data_id_shuffle.py
# 打乱数据集的顺序
# 输出打乱后的顺序
# ****************************************

num_nn = GL_set.num_nn
save_path = GL_set.save_path

# num_nn是小csv的个数
# train_id是维度为num_nn的数组，[0, 1, 2, ..., num_nn-1]
train_id = [ii for ii in range(num_nn)]
# 打乱train_id
np.random.shuffle(train_id)
# 保存打乱后的train_id
# 保存路径
save_file_shuffle_id = save_path + 'shuffle_id.csv'
# 先转化为DataFrame格式，再保存
df = pd.DataFrame(train_id)
df.to_csv(save_file_shuffle_id, encoding='utf-8', index=False, header=False)
print("shuffle id successfully saved")
print("shuffle id path: ", save_file_shuffle_id)
print("end")

# **********补充说明**********
# 举例说明：
# 93个csv算例，每个算例20个时间步，拆成93*20=1860个小csv，那么num_nn=1860
# 那么是维度为1860的数组，[0, 1, 2, ..., 1859]
# 打乱后，可能：train_id=[108, 7, 45, 816, ..., 1392, 648]
# 在之后的训练中，如果训练集大小为1800，那么获取train_id前1800个作为训练集，剩下的60个作为测试集
# 也就是让：data_nn_108, data_nn_7, data_nn_45, data_nn_816, ... 作为训练集
# 剩下：..., data_nn_1392, data_nn_648 作为测试集
