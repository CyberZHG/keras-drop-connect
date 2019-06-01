import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_drop_connect.backend import keras, EAGER_MODE
from keras_drop_connect import DropConnect


class TestDropConnect(TestCase):

    @staticmethod
    def _gen_dense_data(num, w=None):
        x = np.random.standard_normal((num, 5))
        if w is None:
            w = np.random.standard_normal((5, 2))
        y = np.dot(x, w)
        noise = np.random.standard_normal((num, 2)) / 100.0
        y = (y + noise).argmax(axis=-1)
        return x, y, w

    @staticmethod
    def _gen_embed_data(num, w=None):
        x = np.random.randint(0, 10, (num, 10))
        if w is None:
            w = np.random.randint(0, 10, (2,))
        y = np.array([1 if w[0] in x[i] and w[1] in x[i] else 0 for i in range(num)])
        return x, y, w

    def _test_fit_model(self, model, generator=None):
        if generator is None:
            generator = self._gen_dense_data
        x, y, w = generator(65536)
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.fit(x, y,
                  epochs=10,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=1e-3)])
        if not EAGER_MODE:
            model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_connect_%f.h5' % np.random.random())
            model.save(model_path)
            model = keras.models.load_model(model_path, custom_objects={'DropConnect': DropConnect})
        x, y, _ = generator(1024, w)
        predicted = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.not_equal(y, predicted)), 100)

    def test_drop_all(self):
        model = keras.models.Sequential()
        model.add(DropConnect(
            keras.layers.Dense(units=5, activation='tanh'),
            rate=0.2,
            input_shape=(5,)
        ))
        model.add(DropConnect(
            keras.layers.Dense(units=2, activation='softmax'),
            rate=0.2,
        ))
        self._test_fit_model(model)

    def test_drop_dict(self):
        model = keras.models.Sequential()
        model.add(DropConnect(
            keras.layers.Dense(units=5, activation='tanh'),
            rate={'kernel': 0.2},
            input_shape=(5,)
        ))
        model.add(DropConnect(
            keras.layers.Dense(units=2, activation='softmax'),
            rate={'bias': 0.2},
        ))
        self._test_fit_model(model)

    def test_drop_batch_norm(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=5, activation='tanh', input_shape=(5,)))
        model.add(DropConnect(
            keras.layers.BatchNormalization(),
            rate=0.1,
        ))
        model.add(keras.layers.Dense(units=2, activation='softmax'))
        self._test_fit_model(model)

    def test_drop_rnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=10, output_dim=5, mask_zero=True, input_shape=(10,)))
        model.add(DropConnect(
            keras.layers.Bidirectional(keras.layers.GRU(units=2, unroll=True)),
            rate=0.1,
        ))
        model.add(keras.layers.Dense(units=2, activation='softmax'))
        self._test_fit_model(model, generator=self._gen_embed_data)
