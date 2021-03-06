# LF2_RL
A testing little fighter gym simulator for reinforcement learning studying.

## Demo
![](demo_img/demo3.gif)
![](demo_img/demo2.gif)


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

def main():

    lf2_env = gym.make('LittleFighter2-v0')
    lf2_env.reset()

    done = False
    while not done:
        obs, reward, done, info = lf2_env.step(lf2_env.action_space.sample())
        if done:
            lf2_env.reset()
    
    lf2_env.close()

if __name__ == '__main__':
    main()
```

## Action Space
Value | Action | Value | Action
--- | --- | --- | ---
0 | idle    | 8 | run
1 | up      | 9 | combo attack1
2 | down    | 10| combo attack2
3 | left    | 11| combo attack3
4 | right   | 12| combo attack4
5 | A       | 13| combo attack5
6 | J       | 14| combo attack6
7 | D

## Observation Space
Mode | Ob space
---|---
picture| [img_h, img_w, number of stacks]
info|[my mp, my hp, my x, my y, my z, enemy1 x, enemy1 y, enemy1 z]
mix | dict(Game_Screen: picture, Info: info)

## Parameters
Parameter|Description|Default Value
---|---|---
windows_name|window's name|'Little Fighter 2'
player_id|AI player id| 1
down_scale|screenshot downscale| 2
frame_stack| number of frames to stack| 4
frame_skip| number of frames to skip between each frame| 1
reset_skip_sec|immortal time when each round begins| 2
mode| observation mode| 'mix'


## Notice
*  Before training/testing, setup your gamemode to "VS Mode" and select your character first.
* Please ALWAYS put lf2 windows on top, otherwise you may result in random words typed in your focused window.(May be fixed in future updates.) 