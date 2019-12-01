from LF2_State import LF2_State
import cv2


def main():

    lf2_stat = LF2_State('Little Fighter 2', windows_scale=1.25)

    while True:
        img, player = lf2_stat.get_state(id=1)
        print('x: {}, y: {}, z: {}'.format(player.x_pos, player.y_pos, player.z_pos))
        if img is not None:
            cv2.imshow('img', img)
            cv2.waitKey(2)

if __name__ == '__main__':
    main()

