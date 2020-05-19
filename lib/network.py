"""
DNN Class
@Author Chanwoo Kwon Yonsei univ. researcher. since 2020.05~
수정목록
- tensorflow 버전의 클래스 추가(DNN_tf), 2020.05.18, Chanwoo Kwon
"""

import numpy as np
import tensorflow as tf


class DNN:

    @staticmethod
    def reLU(value):
        """
        Rectified Linear Unit activation (0 if lower than 0)
        :param value: numpy array data
        :return: activated numpy array data
        """
        return value * (value > 0)

    @staticmethod
    def strarr_to_float(strarr):
        """
        change string array to float array
        :param strarr: string array
        :return: float array
        """
        floatarr = []
        for s in strarr:
            floatarr.append(float(s))

        return floatarr

    def __init__(self, main_data_path):
        self._main_data_path = main_data_path
        self._input_layer = []  # 3 input values
        self._layers = [[], [], [], [], [], [], []]  # 7 layers (6 hidden layers and output layer)
        self._edges = [[], [], [], [], [], [], []]  # 7 edges
        self._output_layer = np.zeros([1])  # output is 1 dimension

        self._test_data = []
        self._train_data = []

    def load_file(self):
        with open('{}/testset-wonju_ldl.txt'.format(self._main_data_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.replace('\r\n', '').split('\t')
                self._test_data.append({'answer': splits[3], 'x': DNN.strarr_to_float(splits[:3])})

        with open('{}/ldl-total-2020-05-17-2.txt'.format(self._main_data_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.replace('\r\n', '').split('\t')
                self._train_data.append({'answer': splits[3], 'x': DNN.strarr_to_float(splits[:3])})

        for i in range(len(self._edges)):
            with open('./data/weight{}.csv'.format(i + 1)) as f:
                lines = f.readlines()
                for line in lines:
                    self._edges[i].append(np.fromstring(line.replace('\r\n', ''), dtype=np.float64, sep=','))

        for i in range(len(self._layers)):
            with open('./data/bias{}.csv'.format(i + 1)) as f:
                lines = f.readlines()
                arr = []
                for line in lines:
                    arr.append(line)

                self._layers[i] = np.array(arr).astype(np.float64)

    def cal_layer(self, input, level):
        layer = np.array([], dtype=np.float)
        for i in range(len(self._edges[level])):
            input_np = np.array(input)
            edge_np = np.array(self._edges[level][i])
            layer = np.concatenate((layer, [(input_np * edge_np).sum()]), axis=0)

        if level < len(self._layers) - 1:
            return DNN.reLU(layer + self._layers[level])

        return layer + self._layers[level]

    def predict(self, input):
        layers = [np.array(input['x'])]

        for i in range(len(self._layers)):
            layers.append(self.cal_layer(layers[i], i))

        return layers[-1][0]

    def validate(self):
        for i in range(len(self._train_data)):
            predict = self.predict(self._train_data[i])
            print(predict)
            if i == 10:
                break

    def test(self):
        for i in range(len(self._test_data)):
            predict = self.predict(self._test_data[i])
            print(predict)
            break


class DNN_tf:

    def __init__(self, main_data_path):
        self._main_data_path = main_data_path
        self._model = None
        self._train_data = [[], []]
        self._test_data = [[], []]
        self.load_file()
        self.build_model()

    def load_file(self):
        with open('{}/ldl-total-2020-05-17-2.txt'.format(self._main_data_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.replace('\r\n', '').split('\t')
                self._train_data[0].append([float(splits[3])])
                self._train_data[1].append(DNN.strarr_to_float(splits[:3]))

        with open('{}/testset-wonju_ldl.txt'.format(self._main_data_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.replace('\r\n', '').split('\t')
                self._test_data[0].append([float(splits[3])])
                self._test_data[1].append(DNN.strarr_to_float(splits[:3]))

    def build_model(self):
        inputs = tf.keras.layers.Input([3], dtype=tf.float64)
        hidden = inputs
        for i in range(6):
            hidden = tf.keras.layers.Dense(30, activation=tf.nn.relu)(hidden)
        outputs = tf.keras.layers.Dense(1)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.load_weights('{}/checkpoints/cholesterol_9100.tf'.format(self._main_data_path))

    def predict(self, input):
        return self._model(np.array([input['x']]), training=False).numpy()[0][0]

    def test(self):
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        res = self._model(np.array(self._test_data[1]), training=False)

        loss = tf.keras.losses.MSE(tf.convert_to_tensor(self._test_data[0]), res)

        avg_loss.update_state(loss)

        print(avg_loss.result().numpy())

        avg_loss.reset_states()
