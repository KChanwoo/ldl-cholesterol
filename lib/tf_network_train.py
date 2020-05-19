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

    train_data = [[], []]
    test_data = [[], []]
    with open('{}/ldl-total-2020-05-17-2.txt'.format('../data'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.replace('\r\n', '').split('\t')
            train_data[0].append([float(splits[3])])
            train_data[1].append(DNN.strarr_to_float(splits[:3]))

    with open('{}/testset-wonju_ldl.txt'.format('../data'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.replace('\r\n', '').split('\t')
            test_data[0].append([float(splits[3])])
            test_data[1].append(DNN.strarr_to_float(splits[:3]))

    # set training variable

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    epoch = 20000
    batch_size = 1000
    iter = round(len(train_data[0]) / batch_size)

    for i in range(epoch):
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        for j in range(iter):
            labels = train_data[0][j * batch_size: j * batch_size + batch_size]
            inputs = train_data[1][j * batch_size: j * batch_size + batch_size]
            with tf.GradientTape() as tape:
                outputs = model(np.array(inputs), training=True)
                loss = tf.keras.losses.MSE(tf.convert_to_tensor(labels), outputs)  # calculate loss using MSE

            grads = tape.gradient(loss, model.trainable_variables)  # calculate gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update gradients

            avg_loss.update_state(loss)

        avg_loss_value = avg_loss.result().numpy()
        print('Epoch: {} Cost: {}'.format(i, avg_loss_value))

        # save weight every 100 epochs
        if i % 100 == 0:
            model.save_weights('checkpoints/cholesterol_{}.tf'.format(i))

        avg_loss.reset_states()

