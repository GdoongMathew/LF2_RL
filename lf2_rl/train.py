import numpy as np
# import torch
import gym
import lf2_gym
import cv2

from lf2_rl.Model_keras import DQN

if __name__ == '__main__':

    mode = 'mix'
    karg = dict(frame_stack=3, frame_skip=1, reset_skip_sec=2, mode=mode)
    lf2_env = gym.make('LittleFighter2-v0', **karg)

    act_n = lf2_env.action_space.n
    state_n = [lf2_env.observation_space['Game_Screen'].shape,
               lf2_env.observation_space['Info'].shape[0]]

    # obs = [obs['Game_Screen'], obs['Info']]
    train_ep = 100000
    agent = DQN(act_n, state_n, 0,
                memory_capacity=60000,
                batch_size=60,
                learning_rate=0.00001,
                momentum=0.8,
                save_freq=200,
                epsilon=0.85,
                dueling=True,
                prioritized=True)
    records = []
    for ep in range(train_ep):

        # pass episode to tensorboard.
        agent.tb.step = ep
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
        observation = DQN.trans_obser(pic, info, mode)

        iter_cnt, total_reward = 0, 0

        while 1:
            iter_cnt += 1

            lf2_env.render(mode='console')

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
            observation_ = DQN.trans_obser(pic, info, mode)
            # RL learn from this transition
            agent.store_transition(observation, action, reward, observation_)
            total_reward += reward

            if done:
                total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                print("Episode {} finished after {} timesteps, total reward is {}".format(ep + 1, iter_cnt,
                                                                                          total_reward))

                if agent.memory_counter > agent.memory_capacity:
                    for i in range(15):
                        agent.learn()
                    print('RL learned 15 times.')
                break

    print('-------------------------')
    print('Finished training')
    lf2_env.close()
    print('Saving model.')
    agent.save_weight('./model')

