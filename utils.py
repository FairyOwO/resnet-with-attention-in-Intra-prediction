import numpy as np
import cv2
import os

def read_im_cv2(path):
    img = cv2.imread(path)
    return img


def draw_imgs(imgs, area: int):
    for j, i in enumerate(imgs):
        if not os.path.exists('./draw_imgs/'+str(j)):
            os.mkdir('./draw_imgs/'+str(j))
        for m in range(i.shape[0]//area):
            for n in range(i.shape[1]//area):
                try:
                    a = i[m*area:(m+1)*area,n*area:(n+1)*area]
                    cv2.imwrite('./draw_imgs/'+str(j)+'/'+str(m)+'_'+str(n) +'.png', a)
                    # print(a)
                except:
                    pass

def zoning(img, n: int):
    if n == 0:
        assert 'error'
    else:
        img = np.array(img)
        x = np.concatenate((img[:n, :], img[n:, :n].swapaxes(0, 1)), axis=1)
        y = img[n:, n:]
        return x, y



if __name__ == '__main__':
    pass
    # draw_imgs([read_im_cv2(r'./imags/@4LH60%[2RAFAJ3R3}82YN4.png')], 32)
    x, y = zoning(read_im_cv2(r'./imags/{ERQP8DHG4HJ37]G_74O]EC.png'), 100)
    print(x.shape, y.shape)