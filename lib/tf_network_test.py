import tensorflow as tf
import numpy as np

from lib.network import DNN

with tf.device('/cpu:0'):
    # build model
    inputs = tf.keras.layers.Input([3], dtype=tf.float64)
    hidden = inputs
    for i in range(6):
        hidden = tf.keras.layers.Dense(30, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(1)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # load data

    test_data = [[], []]
    with open('{}/ldl-total-2020-05-17-2.txt'.format('../data'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.replace('\r\n', '').split('\t')
            test_data[0].append([float(splits[3])])
            test_data[1].append(DNN.strarr_to_float(splits[:3]))
            if len(test_data[0]) == 100:
                break

    # set testing variable
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    min_avg_loss = 100000
    min_index = -1

    # dead code after here
    # 9100 epoch point is best cost (55.38893 -> train_data, 70.55025 -> test_data)
    for i in range(200):
        model.load_weights('checkpoints/cholesterol_{}.tf'.format(i * 100))

        res = model(np.array(test_data[1]), training=False)

        loss = tf.keras.losses.MSE(tf.convert_to_tensor(test_data[0]), res)

        avg_loss.update_state(loss)

        avg_loss_value = avg_loss.result().numpy()
        if min_avg_loss > avg_loss_value:
            min_avg_loss = avg_loss_value
            min_index = i

        avg_loss.reset_states()

        model.reset_states()

    model.load_weights('checkpoints/cholesterol_{}.tf'.format(min_index * 100))
    res = model(np.array(test_data[1]), training=False)
    print(min_avg_loss)
    print(min_index)
    print(tf.sqrt(min_avg_loss))
    print(res)
