import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from .basemodel import BaseModel
from .util import Memory

# from pytorch_memlab import profile_every

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, action_n, state_n, batch_size=32, lstm_hidden=50, dueling=False):
        super(Net, self).__init__()

        picture_n, feature_n = state_n[0], state_n[1]
        # input 4 x 240 x 500
        # 492, 232
        self.conv1 = nn.Conv2d(picture_n[-1], 32, kernel_size=8, stride=4, dilation=2).to(device)  # 32 x 57 x 122
        self.bn1 = nn.BatchNorm2d(32).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=5).to(device)        # 64 x 12 x 25
        self.bn2 = nn.BatchNorm2d(64).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1).to(device)        # 64 x 8 x 21
        self.bn3 = nn.BatchNorm2d(64).to(device)

        self.batch_size = batch_size

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(picture_n[0], kernel_size=8, stride=4),
                        kernel_size=2, stride=5),
                    kernel_size=5, stride=1)

        convh = conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(picture_n[1], kernel_size=8, stride=4),
                        kernel_size=2, stride=5),
                    kernel_size=5, stride=1)

        self.fc1 = nn.Linear(64 * convw * convh, 512).to(device)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(512 + feature_n, 100).to(device)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

        # LSTM
        self.lstm_hidden = lstm_hidden
        self.action_n = action_n

        self.lstm = nn.LSTMCell(100, self.lstm_hidden, bias=True).to(device)
        self.hx = torch.randn(1, self.lstm_hidden).cuda()
        self.cx = torch.randn(1, self.lstm_hidden).cuda()

        self.dueling = dueling
        if dueling:
            self.adv = nn.Linear(self.lstm_hidden, self.action_n).to(device)
            self.val = nn.Linear(self.lstm_hidden, 1).to(device)

        else:
            self.out = nn.Linear(self.lstm_hidden, self.action_n).to(device)
            self.out.weight.data.normal_(0, 0.1)  # initialization

    # @ profile_every(1)
    def forward(self, x):
        img, feature = x[0], x[1]
        # img = cv2.resize(img, (240, 500))
        img = Func.relu(self.bn1(self.conv1(img)))
        img = Func.relu(self.bn2(self.conv2(img)))
        img = Func.relu(self.bn3(self.conv3(img)))
        img = img.view(img.size(0), -1)
        img = Func.relu(self.fc1(img))

        # feature nn
        x = torch.cat((img, feature), 1)
        x = Func.relu(self.fc2(x))

        size = x.shape[0]
        if size == 1:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
            x = self.hx
        else:
            x, _ = self.lstm(x, (torch.cat([self.hx] * self.batch_size, 0),
                                 torch.cat([self.cx] * self.batch_size, 0)))
        if self.dueling:
            val = self.val(x)
            adv = self.adv(x)
            x = val + adv - adv.mean(1).unsqueeze(1).expand(self.batch_size, self.action_n)
        else:
            x = self.out(x)
        return x


class DQN(BaseModel):
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9, epsilon=0.5,
                 memory_capacity=20000, batch_size=32, update_freq=100, lstm_hidden=50, dueling=False, prioritized=False):

        super(DQN, self).__init__(action_n, state_n, env_shape, learning_rate=learning_rate, reward_decay=reward_decay,
                                  epsilon=epsilon,
                                  memory_capacity=memory_capacity, batch_size=batch_size, update_freq=update_freq,
                                  prioritized=prioritized)

        self.eval_net = Net(self.action_n, self.state_n, batch_size=self.batch_size, lstm_hidden=lstm_hidden,
                            dueling=dueling)
        self.target_net = Net(action_n, state_n, batch_size=batch_size, lstm_hidden=lstm_hidden, dueling=dueling)

        if self.prioritized:
            self.memory = Memory(self.memory_capacity)
        else:
            self.memory = np.zeros((self.memory_capacity, (self.state_n[0][2] * self.state_n[0][0] * self.state_n[0][1] + self.state_n[1]) * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    # @ profile_every(5)
    def choose_action(self, x):
        x = [torch.unsqueeze(torch.FloatTensor(x[0]).to(device), 0),
             torch.unsqueeze(torch.FloatTensor(x[1]).to(device), 0)]

        if np.random.uniform() < self.epsilon:
            action_val = self.eval_net.forward(x).cpu()
            action = torch.max(action_val, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)

        return action

    def learn(self):
        # target parameter update
        if self.step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.step_counter += 1
        # sample batch transitions

        if self.prioritized:
            tree_idx, b_memory, is_weight = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_capacity, self.batch_size)
            b_memory = self.memory[sample_index, :]
        picture_idx = self.state_n[0][0] * self.state_n[0][1] * self.state_n[0][2]
        feature_idx = self.state_n[1]
        state_idx = picture_idx + feature_idx  # 4 * 160 * 380 + 28
        b_picture = torch.FloatTensor(b_memory[:, :picture_idx]).to(device)
        b_feature = torch.FloatTensor(b_memory[:, picture_idx:state_idx]).to(device)
        b_a = torch.LongTensor(b_memory[:, state_idx:state_idx + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, state_idx + 1:state_idx + 2]).to(device)
        b_picture_ = torch.FloatTensor(b_memory[:, -state_idx:-feature_idx]).to(device)
        b_feature_ = torch.FloatTensor(b_memory[:, -feature_idx:]).to(device)
        # reshape (batch_size, 4, 160, 380)

        b_picture = torch.reshape(b_picture, (self.batch_size, self.state_n[0][2], self.state_n[0][0], self.state_n[0][1])).to(device)
        b_picture_ = torch.reshape(b_picture_, (self.batch_size, self.state_n[0][2], self.state_n[0][0], self.state_n[0][1])).to(device)
        b_s = [b_picture, b_feature]
        b_s_ = [b_picture_, b_feature_]

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()

        if self.backward_count > 0:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
            self.backward_count += 1
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(),
                   'model/DQN/mix_eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))
        torch.save(self.target_net.state_dict(),
                   'model/DQN/mix_target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))

    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name))

