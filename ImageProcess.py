from LF2_State import LF2Env
import numpy as np
import cv2


def main():

    lf2_env = LF2Env('Little Fighter 2', windows_scale=1.25)

    while True:
        img, player = lf2_env.get_state(player_id=1)

        action = player.get_action_list()

        act_id = np.random.randint(len(action))

        lf2_env.step(act_id)

        if img is not None:
            cv2.imshow('img', img)
            cv2.waitKey(2)

        if not player.is_alive:
            lf2_env.reset()
            print('reset')


if __name__ == '__main__':
    main()

