"""
Compare DNN LDL Cholesterol MSE, The Friedelwald equation MSE
@Author Chanwoo Kwon, Yonsei Univ. Researcher. since 2020.05~
"""

from lib.network import DNN, DNN_tf
import tensorflow as tf
import matplotlib.pyplot as plt

test_data = [[], []]
data_for_graph = []

with open('{}/testset-wonju_ldl.txt'.format('./data'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        splits = line.replace('\r\n', '').split('\t')
        test_data[0].append([float(splits[3])])
        data_for_graph.append(float(splits[3]))
        test_data[1].append(DNN.strarr_to_float(splits[:3]))

# the Friedelwald equation : total cholesterol - HDL choesterol - triglyceride / 5

ldl_c_f = []
ldl_c_f_for_graph = []
for i in range(len(test_data[0])):
    data = test_data[1][i]
    total_c = data[0]
    hdl_c = data[1]
    tri = data[2]

    ldl_c = total_c - hdl_c - tri / 5
    ldl_c_f.append([ldl_c])
    ldl_c_f_for_graph.append(ldl_c)

loss_f = tf.keras.losses.MSE(tf.convert_to_tensor(test_data[0]), tf.convert_to_tensor(ldl_c_f))
avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
avg_loss.update_state(loss_f)

print(avg_loss.result().numpy())

avg_loss.reset_states()

# using DNN

dnn = DNN_tf('./data')

loss, res = dnn.test()
print(loss)
ldl_c_d_for_graph = []
res = res.numpy()
for i in range(len(res)):
    ldl_c_d_for_graph.append(res[i][0])

# multiple line plot
plt.plot(range(len(test_data[0])), ldl_c_f_for_graph, marker='', color='royalblue', linewidth=2, label="Friedewald equation")
plt.plot(range(len(test_data[0])), ldl_c_d_for_graph, marker='', color='teal', linewidth=2, label="DNN Regression")
plt.plot(range(len(test_data[0])), data_for_graph, marker='', color='coral', linewidth=2, label="Measured LDL")
plt.legend()
plt.show()

