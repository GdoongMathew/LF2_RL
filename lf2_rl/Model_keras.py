# referenced from https://github.com/keras-rl/keras-rl
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


from .util import Memory, ModifiedTensorBoard
from .basemodel import BaseModel
from glob import glob
import numpy as np
import os

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


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


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

        model = Conv2D(32, (3, 3), strides=2)(img_input)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        model = Conv2D(64, (2, 2), strides=2)(model)
        model = BatchNormalization()(model)
        # model = Activation('relu')(model)

        model = xception_block(model, [64, 64, 64], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False)

        model = xception_block(model, [128, 128, 128], 'entry_flow_block1',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False)

        model = xception_block(model, [182, 182, 182], 'entry_flow_block3',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False)

        model = xception_block(model, [182, 256, 256], 'exit_flow_block1',
                               skip_connection_type='conv', stride=1, rate=4,
                               depth_activation=True)

        model = xception_block(model, [384, 384, 512], 'exit_flow_block2',
                               skip_connection_type='none', stride=1, rate=4,
                               depth_activation=True)

        # Flatten before Dense layers
        model = Flatten()(model)
        model = Dense(512, activation=tf.nn.relu)(model)
        model = Concatenate()([model, fea_input])
        model = Dense(256, activation=tf.nn.relu)(model)

        if self.dueling:
            model = Dense(self.action_n + 1, activation='linear')(model)

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
                 weight_path=f'./Keras_Save/keras_dqn.h5', lstm_hidden=50, **kwargs):
        super(DQN, self).__init__(action_n, state_n, env_shape, **kwargs)

        self.dueling = dueling

        self.eval_net = Net(self.action_n, self.state_n,
                            SGD(lr=self.lr, momentum=self.momentum),
                            batch_size=self.batch_size,
                            lstm_hidden=lstm_hidden,
                            dueling=self.dueling,
                            duel_type=duel_type).create_model()
        self.eval_net.summary()
        self.target_net = Net(self.action_n, self.state_n,
                              SGD(lr=self.lr, momentum=self.momentum),
                              batch_size=self.batch_size,
                              lstm_hidden=lstm_hidden,
                              dueling=self.dueling,
                              duel_type=duel_type).create_model()
        self.target_net.set_weights(self.eval_net.get_weights())
        if isinstance(weight_path, type(None)):
            path = glob(f'./Keras_Save/keras_dqn_*.h5')
            self.weight_path = path[0]
        elif not isinstance(weight_path, type(None)) and isinstance(weight_path, str):
            self.weight_path = weight_path
            if os.path.isfile(self.weight_path):
                print(f'Restoring Weight from {self.weight_path}')
                self.restore_weight(self.weight_path)

        self.prioritized = prioritized
        tb_path = os.path.join(os.path.dirname(self.weight_path), 'Tensorboard')
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        self.tb = ModifiedTensorBoard(log_dir=tb_path)

        # self.compile(SGD(lr=self.lr, momentum=self.momentum))
        self.tb.set_model(self.eval_net)

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
        if np.random.random() > self.policy():
            action_val = self.target_net.predict_on_batch([[x[0]], [x[1]]])
            action = np.argmax(action_val, axis=-1)[0]
            # print(f'Action_V: {action_val}, action_id: {action}')

        else:
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def policy(self):
        p = self.epsilon * (self.epsilon_decay ** self.step_counter)
        return max(p, 0.001)

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
        b_done = []

        for mem in b_memory:
            b_picture.append(mem.state_0[0])
            b_feature.append(mem.state_0[1])
            b_a.append(mem.action)
            b_r.append(mem.reward)
            b_done.append(mem.done)
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

        target_q = self.target_net.predict_on_batch(b_s_)
        if self.dueling:
            q_val = self.eval_net.predict_on_batch(b_s)
            actions = np.argmax(q_val, axis=1)
            q_batch = target_q[range(self.batch_size), actions]
        else:
            q_batch = np.max(target_q, axis=1).flatten()

        targets = np.zeros((self.batch_size, self.action_n))
        dummy_targets = np.zeros((self.batch_size, ))
        masks = np.zeros((self.batch_size, self.action_n))

        for idx, (target, mask, q, action, done) in enumerate(zip(targets, masks, q_batch, b_a, b_done)):
            if not done:
                R = q * self.gamma + b_r[idx]
            else:
                R = q
            target[action] = R
            mask[action] = 1.0  # calculate loss of this action
            dummy_targets[idx] = R

        ins = [b_s] if type(self.eval_net.input) is not list else b_s
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]
        self.trainable_model.fit(ins + [targets, masks], [dummy_targets, targets],
                                 batch_size=self.batch_size,
                                 verbose=True,
                                 callbacks=[self.tb])
        # print(f'metrics: {metrics}')
        if self.step_counter % self.update_freq == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
        self.step_counter += 1
        self.tb.step += 1

        if self.step_counter % self.save_freq:
            self.save_weight(self.weight_path)

    def learn2(self):
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
        b_done = []
        b_picture_ = []
        b_feature_ = []

        for mem in b_memory:
            b_picture.append(mem.state_0[0])
            b_feature.append(mem.state_0[1])
            b_a.append(mem.action)
            b_r.append(mem.reward)
            b_done.append(mem.done)
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

        target_q = self.target_net.predict_on_batch(b_s_)
        q_val = self.eval_net.predict_on_batch(b_s)

        for i in range(self.batch_size):
            rs = b_r[i]
            if not b_done[i]:
                rs += self.gamma * np.max(target_q[i])
            q_val[i][b_a[i]] = rs

        self.eval_net.fit(b_s, q_val, batch_size=self.batch_size, verbose=1, shuffle=False, callbacks=[self.tb])

        if self.step_counter % self.update_freq == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
        self.step_counter += 1
        self.tb.step = self.step_counter

        if self.step_counter % self.save_freq:
            split_txt = os.path.splitext(self.weight_path)
            self.save_weight(split_txt[0] + f'_{self.step_counter}' + split_txt[-1])

    @ staticmethod
    def reward_modify(r):
        return r / 1.

    def save_weight(self, path=None, overwrite=True):
        self.eval_net.save_weights(path, overwrite=overwrite)

    def restore_weight(self, path=None):
        self.eval_net.load_weights(path)
        self.target_net.set_weights(self.eval_net.get_weights())


if __name__ == '__main__':
    dqn = DQN(10, [(255, 255, 3), (10, )], (1000, 1000), batch_size=10)



