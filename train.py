from model import ResNet
from utils import zoning, read_im_cv2
import numpy as np
import tqdm
import torch


def train():
    save_every = 10
    step = 0
    in_channels = 3
    conv_channels = 16
    num_blocks = 10
    n = 8
    device = torch.device('cuda')
    batch_size = 128
    model = ResNet(in_channels, conv_channels, num_blocks, n).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    pb = tqdm(total=save_every, desc='train', initial=step % save_every)
    for epoch in range(save_every):



if __name__ == '__main__':
    n = 8
    x, y = zoning(read_im_cv2(r'./draw_imgs/1/0_0.png'), n)
    print(x.shape)
    print(y.shape)
    model = ResNet(3, 16, 10, n)
    x = torch.FloatTensor([x])
    y = torch.FloatTensor([y])
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    print(model(x))
