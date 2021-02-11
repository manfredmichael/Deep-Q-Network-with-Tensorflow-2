import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense

class Model(tf.keras.Model):
    def __init__(self, obs_shape, act_shape, hidden_layers):
        super(Model, self).__init__()
        self.input_ = Flatten(input_shape=obs_shape)
        self.output = Dense(act_shape, activation='linear')

        self.hidden = []
        for hidden_unit in hidden_layers:
            self.hidden.append(Dense(hidden_unit, activation='relu', kernel_initializer='glorot_uniform'))

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.input_(inputs)
        for hidden in self.hidden:
            x = hidden(x)
        x = self.output(x)
        return x
        