"""
@Author Chanwoo Kwon, Yonsei Univ. Researcher since 2020.05~
"""
from flask import Flask, jsonify, request
from lib.network import DNN, DNN_tf

app = Flask(__name__)

dnn_tf = DNN_tf('./data')


@app.route("/")
def hello():
    return "Hello World"


@app.route("/predict/", methods=['POST'])
def predict():
    value = request.get_json()
    print(value)
    predict = dnn_tf.predict({'x': [value['total'], value['hdl'], value['tri']]})

    return jsonify({"predict": str(predict)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
