# Keras BERT

[![Travis](https://travis-ci.org/CyberZHG/keras-drop-connect.svg)](https://travis-ci.org/CyberZHG/keras-drop-connect)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-drop-connect/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-drop-connect)
[![Version](https://img.shields.io/pypi/v/keras-drop-connect.svg)](https://pypi.org/project/keras-drop-connect/)
![Downloads](https://img.shields.io/pypi/dm/keras-drop-connect.svg)
![License](https://img.shields.io/pypi/l/keras-drop-connect.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-bert/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-bert/blob/master/README.md)\]

## 安装

```bash
pip install keras-drop-connect
```

## 使用

为所有可训练参数设置相同的drop connect概率：

```python
import keras
from keras_drop_connect import DropConnect

DropConnect(
    layer=keras.layers.Dense(units=10),
    rate=0.2,
)
```

通过词典指定参数设置概率：

```python
import keras
from keras_drop_connect import DropConnect

DropConnect(
    layer=keras.layers.Dense(units=10),
    rate={'kernel': 0.2},
)
```
