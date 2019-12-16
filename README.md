# LF2_RL
A testing little fighter gym simulator for reinforcement learning studying.

## Installation
1. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.
```
pip install gym
```
2. Download and install [LF2_RL](https://github.com/GdoongMathew/LF2_RL)
```
git clone https://github.com/GdoongMathew/LF2_RL.git
cd LF2_RL
python setup.py install
```

## Running
```python
import gym
import time

def main():

    lf2_env = gym.make('LittleFighter2-v0')
    now = time.time()
    lf2_env.reset()

    done = False
    while not done:
        obs, reward, done, info = lf2_env.step(lf2_env.action_space.sample())
        if lf2_env.game_over:
            lf2_env.reset()
        if time.time() - now >= 120:
            done = True
    
    lf2_env.close()

if __name__ == '__main__':
    main()
```

## Action Space
Value | Action | Value | Action
--- | --- | --- | ---
0 | idle    | 8 | run
1 | up      | 9 | combo attact1
2 | down    | 10| combo attact2
3 | left    | 11| combo attact3
4 | right   | 12| combo attact4
5 | A       | 13| combo attact5
6 | J       | 14| combo attact6
7 | D

## Notice
*  Before running the script, setup your gamemode to "VS Mode" and select your character first.
* Please ALWAYS put lf2 windows on top, otherwise you may result in random words typed in your focused window.(May be fixed in future updates.) 