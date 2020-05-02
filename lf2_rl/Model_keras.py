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

from keras.optimizers import Adam, SGD
from keras import backend as K


from .util import Memory
from .basemodel import BaseModel
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0.

    x = y_true - y_pred
    squared_loss = 0.5 * K.square(x)
    if np.isinf(clip_value):
        return squared_loss

    condition = K.abs(x) < clip_value
    linear_loss = clip_value * (K.abs(x) - 0.5 * clip_value)

    return tf.where(condition, squared_loss, linear_loss)


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
    def __init__(self, action_n, state_n, optimizer, batch_size=32, lstm_hidden=50, dueling=False, duel_type='max'):
        self.action_n = action_n
        self.batch_size = batch_size
        self.picture_n, self.feature_n = state_n
        self.lstm_hidden = lstm_hidden
        self.dueling = dueling
        self.duel_type = duel_type
        self.optimizer = optimizer

    def create_model(self):
        img_input = Input(shape=self.picture_n)
        fea_input = Input(shape=(self.feature_n, ))

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
            model = Dense(self.action_n + 1)(model)

            # q_val = value + advantages - avg/max(advantages)
            if self.duel_type == 'avg':
                model = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True),
                    output_shape=(self.action_n,))(model)
            elif self.duel_type == 'max':
                model = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True),
                    output_shape=(self.action_n,))(model)
            else:
                raise AssertionError('duel_type must be either avg or max.')
        else:
            # directly output q value
            model = Dense(self.action_n, activation='linear')(model)

        model = Model(inputs=[img_input, fea_input], outputs=model)
        # model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        return model


class DQN(BaseModel):
    def __init__(self, action_n, state_n, env_shape, dueling=False, duel_type='max', prioritized=False,
                 weight_path=None, lstm_hidden=50, **kwargs):
        super(DQN, self).__init__(action_n, state_n, env_shape, **kwargs)

        self.dueling = dueling

        self.eval_net = Net(self.action_n, self.state_n,
                            SGD(lr=self.lr, momentum=self.momentum),
                            batch_size=self.batch_size,
                            lstm_hidden=lstm_hidden,
                            dueling=self.dueling,
                            duel_type=duel_type).create_model()
        if not isinstance(weight_path, type(None)) and isinstance(weight_path, str):
            self.eval_net.load_weights(self.weigh_path)
        self.eval_net.summary()
        self.target_net = Net(self.action_n, self.state_n,
                              SGD(lr=self.lr, momentum=self.momentum),
                              batch_size=self.batch_size,
                              lstm_hidden=lstm_hidden,
                              dueling=self.dueling,
                              duel_type=duel_type).create_model()
        self.target_net.set_weights(self.eval_net.get_weights())
        self.prioritized = prioritized
        self.compile(SGD(lr=self.lr, momentum=self.momentum))

    @staticmethod
    def trans_obser(observation, feature, mode):
        if mode == 'picture':
            return observation
        elif mode == 'feature':
            observation = feature
        elif mode == 'mix':
            observation = [observation, feature]
        return observation

    def choose_action(self, x):
        if np.random.uniform() < self.epsilon:
            action_val = self.target_net.predict_on_batch([[x[0]], [x[1]]])
            action = np.argmax(action_val, axis=-1)[0]

        else:
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def compile(self, optimizer, metrics=[]):
        def clipped_masked_error(args):
            _true, _pred, _mask = args
            loss = huber_loss(_true, _pred, np.inf)
            loss *= mask
            return K.sum(loss, axis=-1)

        metrics += [mean_q]

        y_pred = self.eval_net.output
        y_true = Input(name='y_true', shape=(self.action_n, ))
        mask = Input(name='mask', shape=(self.action_n, ))
        loss_out = Lambda(clipped_masked_error, output_shape=(1, ), name='loss')([y_true, y_pred, mask])
        ins = [self.eval_net.input] if type(self.eval_net.input) is not list else self.eval_net.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,
            lambda y_true, y_pred: K.zeros_like(y_pred)
        ]

        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

    def learn(self):
        # sample batch transitions
        if isinstance(self.memory, Memory):
            # is prioritized
            tree_idx, b_memory, is_weight = self.memory.sample(self.batch_size)

        else:
            sample_index = np.random.choice(self.memory_capacity, self.batch_size)
            b_memory = self.memory[sample_index]

        b_picture = []
        b_feature = []
        b_a = []
        b_r = []
        b_picture_ = []
        b_feature_ = []

        for mem in b_memory:
            b_picture.append(mem.state_0[0])
            b_feature.append(mem.state_0[1])
            b_a.append(mem.action)
            b_r.append(mem.reward)
            b_picture_.append(mem.state_1[0])
            b_feature_.append(mem.state_1[1])

        b_picture = self.data_process(np.asarray(b_picture))
        b_feature = np.asarray(b_feature)
        b_picture_ = self.data_process(np.asarray(b_picture_))
        b_feature_ = np.asarray(b_feature_)

        b_a = np.asarray(b_a)
        b_r = np.asarray(b_r)

        b_s = [b_picture, b_feature]
        b_s_ = [b_picture_, b_feature_]

        if self.dueling:
            q_val = self.eval_net.predict_on_batch(b_s)
            actions = np.argmax(q_val, axis=1)
            target_q = self.target_net.predict_on_batch(b_s_)
            q_batch = target_q[range(self.batch_size), actions]
        else:
            target_q = self.target_net.predict_on_batch(b_s_)
            q_batch = np.max(target_q, axis=1).flatten()

        targets = np.zeros((self.batch_size, self.action_n))
        dummy_targets = np.zeros((self.batch_size, ))
        masks = np.zeros((self.batch_size, self.action_n))

        discounted_reward = self.gamma * q_batch
        assert discounted_reward.shape == b_r.shape
        rs = b_r + discounted_reward

        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, rs, b_a)):
            target[action] = R
            mask[action] = 1.0  # calculate loss of this action
            dummy_targets[idx] = R

        ins = [b_s] if type(self.eval_net.input) is not list else b_s
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        print(f'metrics: {metrics}')
        if self.step_counter % self.update_freq == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
        self.step_counter += 1

    @ staticmethod
    def reward_modify(r):
        return r / 180.

    def save_weight(self, path=None):
        if isinstance(path, type(None)):
            path = '/Keras_Save/'
        self.eval_net.save(path)

    def restore_weight(self, path):
        self.eval_net.load_weights(path)


if __name__ == '__main__':
    dqn = DQN(10, [(255, 255, 3), (10, )], (1000, 1000), batch_size=10)



