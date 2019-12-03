from LF2_State import LF2Env
import cv2


def main():

    lf2_env = LF2Env('Little Fighter 2', windows_scale=1.25)

    while True:
        img, player = lf2_env.get_state(id=1)
        print(player.is_active)
        print('hp: {}'.format(player.Hp))
        print(lf2_env.reward())
        if img is not None:
            cv2.imshow('img', img)
            cv2.waitKey(2)


if __name__ == '__main__':
    main()

