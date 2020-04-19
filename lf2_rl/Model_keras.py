import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import LSTM
from keras import backend as K

from .util import Memory
from .basemodel import BaseModel
import numpy as np


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x


class Net:
    def __init__(self, action_n, state_n, batch_size=32, lstm_hidden=50, dueling=False):
        self.action_n = action_n
        self.batch_size = batch_size
        self.picture_n, self.feature_n = state_n
        self.lstm_hidden = lstm_hidden
        self.dueling = dueling

    def create_model(self):
        img_input = Input(shape=self.picture_n)
        fea_input = Input(shape=self.feature_n)

        model = Conv2D(32, (8, 8), strides=4)(img_input)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (2, 2), strides=5)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (5, 5), strides=1)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        # Flatten before Dense layers
        model = Flatten()(model)

        model = Dense(512, activation=tf.nn.relu)(model)

        model = Concatenate()([model, fea_input])
        model = Dense(100, activation=tf.nn.relu)(model)

        # model = LSTM(self.lstm_hidden)(model)

        if self.dueling:
            adv = Dense(self.action_n)(model)
            val = Dense(1)(model)
            model = Lambda(val + (adv - K.mean(adv)))(model)
        else:
            model = Dense(self.action_n)(model)

        model = Model(inputs=[img_input, fea_input], outputs=model)
        model.compile(optimizer='Adam', loss='mean_squared_error')
        return model


class DQN(BaseModel):
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9, epsilon=0.5,
                 memory_capacity=20000, batch_size=32, update_freq=100, lstm_hidden=50, dueling=False,
                 prioritized=False):
        super(DQN, self).__init__(action_n, state_n, env_shape, learning_rate=learning_rate, reward_decay=reward_decay,
                                  epsilon=epsilon,
                                  memory_capacity=memory_capacity, batch_size=batch_size, update_freq=update_freq,
                                  prioritized=prioritized)

        self.dueling = dueling
        self.model = Net(action_n, state_n, batch_size=batch_size, lstm_hidden=lstm_hidden).create_model()
        self.model.summary()
        self.target_model = Net(action_n, state_n, batch_size=batch_size, lstm_hidden=lstm_hidden).create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.prioritized = prioritized

        self.memory = Memory(self.memory_capacity) if self.prioritized else \
            np.zero((self.memory_capacity, (self.state_n[0][2] * self.state_n[0][0] * self.state_n[0][1] +
                                            self.state_n[1]) * 2 + 2))

    def choose_action(self, x):

        if np.random.uniform() < self.epsilon:
            action_val = self.target_model.predict_on_batch([x])
            action = np.argmax(action_val, axis=-1)

        else:
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)

        return action


if __name__ == '__main__':
    dqn = DQN(10, [(255, 255, 3), (10, )], (1000, 1000), batch_size=10)



