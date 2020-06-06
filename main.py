"""
Diagnostics project
@Author Chanwoo Kwon Yonsei univ. researcher. since 2020.05~
"""

from lib.network import DNN, DNN_tf

# define variable
# dnn = DNN('./data')

# call dnn
# predict = dnn.predict({'x': [152, 61.756, 30]})  # predict one value
# print(predict)
# dnn.validate()  # validate train-data
# dnn.test()  # test test-data

dnn_tf = DNN_tf('./data')
dnn_tf.show_summary()
predict = dnn_tf.predict({'x': [152, 61.756, 30]})
print(predict)
# dnn_tf.test() # calculate cost from 4520 test data (using mse)
