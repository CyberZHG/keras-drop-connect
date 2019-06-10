from .backend import keras
from .backend import backend as K


class DropConnect(keras.layers.Wrapper):

    def __init__(self, layer, rate=0.0, seed=None, scale=True, **kwargs):
        super(DropConnect, self).__init__(layer, **kwargs)
        if isinstance(rate, dict):
            for name in list(rate.keys()):
                rate[name] = min(1., max(0., rate[name]))
        else:
            rate = min(1., max(0., rate))
        self.rate = rate
        self.seed = seed
        self.scale = scale
        self.supports_masking = self.layer.supports_masking

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if keras.utils.generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if keras.utils.generic_utils.has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask

        def dropped_weight(weight, drop_connect_rate):
            def _dropped_weight():
                dropped = K.dropout(weight, drop_connect_rate, seed=self.seed)
                if self.scale:
                    dropped /= K.constant(1.0 - drop_connect_rate)
                return dropped
            return _dropped_weight

        origins = {}
        if isinstance(self.rate, dict):
            for name, rate in self.rate.items():
                w = getattr(self.layer, name)
                if w in self.layer.trainable_weights:
                    origins[name] = w
                    if 0. < rate < 1.:
                        setattr(self.layer, name, K.in_train_phase(dropped_weight(w, rate), w, training=training))
        else:
            for name in dir(self.layer):
                try:
                    w = getattr(self.layer, name)
                except Exception as e:
                    continue
                if w in self.layer.trainable_weights:
                    origins[name] = w
                    if 0. < self.rate < 1.:
                        setattr(self.layer, name, K.in_train_phase(dropped_weight(w, self.rate), w, training=training))
        outputs = self.layer.call(inputs, **kwargs)
        for name, w in origins.items():
            setattr(self.layer, name, w)
        return outputs

    def get_config(self):
        config = {
            'rate': self.rate,
            'seed': self.seed,
            'scale': self.scale,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
