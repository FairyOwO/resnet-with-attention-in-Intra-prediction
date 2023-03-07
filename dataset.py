from utils import *
import os


def load_training_data(dir):
    imgs = []
    for i in os.listdir(dir):
        imgs.append(read_im_cv2(dir+'/'+i))
    return imgs


if __name__ == '__main__':
    imgs = []
    for i in os.listdir('./imags/'):
        imgs.append(read_im_cv2('./imags/' + i))

    draw_imgs(imgs, 16)
