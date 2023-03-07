from model import ResNet
from utils import zoning, read_im_cv2
from dataset import load_training_data
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def train():
    statefile = './model/model.pth'
    save_every = 100
    step = 0
    in_channels = 3
    conv_channels = 32
    num_blocks = 16
    n = 8
    device = torch.device('cuda')
    batch_size = 128

    raw_data = load_training_data('./draw_imgs/1/')
    temp_x = []
    temp_y = []
    for i in raw_data:
        x, y = zoning(i, n)
        temp_x.append(x)
        temp_y.append(y)
    x = torch.FloatTensor(np.array(temp_x))
    y = torch.FloatTensor(np.array(temp_y))
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    x = x.to(device)
    y = y.to(device)

    train_data = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = ResNet(in_channels, conv_channels, num_blocks, n).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()

    writer = SummaryWriter('./tensorboard_dir/')
    while step < 100000:
        pb = tqdm(total=save_every, desc='train')
        for epoch in range(save_every):
            model.train()
            for data in train_loader:
                x, y = data
                optimizer.zero_grad()
                z = model(x)
                loss = loss_func(z, y)
                loss.backward()
                optimizer.step()
            step += 1
            pb.update(1)
        pb.close()

        writer.add_scalar('train/loss', loss.item(), step)
        writer.flush()
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps': step,
            'timestamp': datetime.now().timestamp()
        }
        torch.save(state, statefile)



if __name__ == '__main__':
    train()
