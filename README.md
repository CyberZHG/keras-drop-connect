# Keras Drop-Connect

[![Travis](https://travis-ci.com/CyberZHG/keras-drop-connect.svg)](https://travis-ci.org/CyberZHG/keras-drop-connect)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-drop-connect/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-drop-connect)

**This repo is outdated and will no longer be maintained.**

## Install

```bash
pip install git+https://github.com/cyberzhg/keras-drop-connect
```

## Usage

Set drop connect rate for all weights:

```python
import keras
from keras_drop_connect import DropConnect

DropConnect(
    layer=keras.layers.Dense(units=10),
    rate=0.2,
)
```

Set drop connect rate with a dict:

```python
import keras
from keras_drop_connect import DropConnect

DropConnect(
    layer=keras.layers.Dense(units=10),
    rate={'kernel': 0.2},
)
```
