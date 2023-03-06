from utils import *
import os

if __name__ == '__main__':
    imgs = []
    for i in os.listdir('./imags/'):
        imgs.append(read_im_cv2('./imags/' + i))

    draw_imgs(imgs, 16)
