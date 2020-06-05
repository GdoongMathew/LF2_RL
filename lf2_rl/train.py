import numpy as np
# import torch
import gym
import lf2_gym
import os
import cv2

from lf2_rl.Model_keras import DQN

if __name__ == '__main__':

    mode = 'mix'
    karg = dict(frame_stack=3, frame_skip=1, reset_skip_sec=2, mode=mode, gray_scale=False, downscale=2)
    lf2_env = gym.make('LittleFighter2-v0', **karg)

    act_n = lf2_env.action_space.n
    state_n = [lf2_env.observation_space['Game_Screen'].shape,
               lf2_env.observation_space['Info'].shape[0]]

    # obs = [obs['Game_Screen'], obs['Info']]
    train_ep = 100000
    agent = DQN(act_n, state_n, 0,
                update_freq=2000,
                # weight_path=f'./Keras_Save/keras_dqn_7079.h5',
                memory_capacity=60,
                batch_size=8,
                learning_rate=1e-6,
                momentum=0.9,
                save_freq=200,
                epsilon=0.995,
                dueling=True,
                prioritized=True)
    records = []

    max_r = 0
    weight_path = f'./Keras_Save/keras_dqn_5174.h5'

    for ep in range(train_ep):

        # pass episode to tensorboard.
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

            # cv2.imshow('img', pic)
            # cv2.waitKey(1)
            observation_ = DQN.trans_obser(pic, info, mode)
            # RL learn from this transition
            agent.store_transition(observation, action, reward, observation_, done)
            total_reward += reward

            observation = observation_

            if done:
                # total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                agent.tb.update_stats(total_reward=total_reward, epsilon=ep)
                print("Episode {} finished after {} timesteps, total reward is {}".format(ep + 1, iter_cnt,
                                                                                          total_reward))

                if total_reward > max_r:
                    split_txt = os.path.splitext(weight_path)
                    agent.save_weight(split_txt[0] + f'_{agent.step_counter}' + split_txt[-1])

                if agent.memory_counter > agent.memory_capacity:
                    for i in range(iter_cnt):
                        agent.learn()
                    print('RL learned 15 times.')
                break

    print('-------------------------')
    print('Finished training')
    lf2_env.close()
    print('Saving model.')
    agent.save_weight('./model')

