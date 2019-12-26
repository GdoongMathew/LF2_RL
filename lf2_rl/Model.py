import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import gym
import lf2_gym
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Net(nn.Module):
    def __init__(self, action_n, state_n):
        super(Net, self).__init__()

        picture_n, feature_n = state_n[0], state_n[1]

        # input 4 x 240 x 500
        # 492, 232
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, dilation=2).to(device)  # 32 x 57 x 122
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=5).to(device)        # 64 x 12 x 25
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1).to(device)        # 64 x 8 x 21

        self.fc1 = nn.Linear(64 * 8 * 21, 200).to(device)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(200 + feature_n, 50).to(device)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, action_n).to(device)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        img, feature = x[0], x[1]
        # img = cv2.resize(img, (240, 500))
        img = Func.relu(self.conv1(img))
        img = Func.relu(self.conv2(img))
        img = Func.relu(self.conv3(img))
        img = img.view(img.size(0), -1)
        img = Func.relu(self.fc1(img))

        # feature nn
        x = torch.cat((img, feature), 1)
        x = Func.relu(self.fc2(x))
        return self.out(x)


class DQN:
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9, epsilon=0.5,
                 memory_capacity=20000, batch_size=32, update_freq=100):
        self.eval_net = Net(action_n, state_n)
        self.target_net = Net(action_n, state_n)
        self.action_n = action_n
        self.state_n = state_n
        self.lr = learning_rate
        self.gamma = reward_decay
        self.env_shape = env_shape
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.update_freq = update_freq

        self.step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, (self.state_n[0][2] * self.state_n[0][0] * self.state_n[0][1] + self.state_n[1]) * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        img = torch.unsqueeze(torch.FloatTensor(x[0]).to(device), 0)
        feature = torch.unsqueeze(torch.FloatTensor(x[1]).to(device), 0)
        x = [img, feature]

        if np.random.uniform() < self.epsilon:
            action_val = self.eval_net.forward(x).cpu()
            action = torch.max(action_val, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)

        return action

    def store_transition(self, s, a, r, s_):
        s = np.append(np.reshape(s[0], -1), s[1])  # [(4, 160, 380), 28] -> 4 * 160 * 380 + 28
        s_ = np.append(np.reshape(s_[0], -1), s_[1])
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        picture_idx = 4 * self.state_n[0][0] * self.state_n[0][1]
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
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(),
                   'model/DQN/mix_eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))
        torch.save(self.target_net.state_dict(),
                   'model/DQN/mix_target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, self.batch_size))

    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name))


def transObser(observation, feature, mode):
    if mode == 'picture':
        observation = np.transpose(observation, (2, 1, 0))
        observation = np.transpose(observation, (0, 2, 1))
    elif mode == 'feature':
        observation = feature
    elif mode == 'mix':
        observation_ = np.transpose(observation, (2, 1, 0))
        observation_ = np.transpose(observation_, (0, 2, 1))
        observation = [observation_, feature]
    return observation


if __name__ == '__main__':

    mode = 'mix'
    karg = dict(frame_stack=4, frame_skip=1, reset_skip_sec=2, mode=mode)
    lf2_env = gym.make('LittleFighter2-v0', **karg)

    act_n = lf2_env.action_space.n
    state_n = [lf2_env.observation_space['Game_Screen'].shape,
               lf2_env.observation_space['Info'].shape[0]]

    # obs = [obs['Game_Screen'], obs['Info']]
    #Y
    train_ep = 600000
    agent = DQN(act_n, state_n, 0)
    records = []
    #
    for ep in range(train_ep):
        obs = lf2_env.reset()

        pic = None
        info = None

        if mode == 'mix':
            pic = obs['Game_Screen']
            info = obs['Info']
        elif mode == 'picture':
            pic = obs
        else:
            info = obs
        observation = transObser(pic, info, mode)

        iter_cnt, total_reward = 0, 0

        while 1:
            iter_cnt += 1

            lf2_env.render()

            # RL choose action based on observation
            action = agent.choose_action(observation)
            # RL take action and get next observation and reward
            obs, reward, done, _ = lf2_env.step(action)
            if mode == 'mix':
                pic = obs['Game_Screen']
                info = obs['Info']
            elif mode == 'picture':
                pic = obs
            else:
                info = obs
            observation_ = transObser(pic, info, mode)
            # RL learn from this transition
            agent.store_transition(observation, action, reward, observation_)
            if agent.memory_counter > agent.memory_capacity:
                agent.learn()

            total_reward += reward

            if done:
                total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                print("Episode {} finished after {} timesteps, total reward is {}".format(ep + 1, iter_cnt,
                                                                                          total_reward))
                break

    print('-------------------------')
    print('Finished training')
    lf2_env.close()
    print('Saving model.')
    agent.save_model()


    #
    # dqn = DQN(act, obs)
    # dqn.forward(obs)
