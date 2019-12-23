import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init(self, action_n, state_n):
        super(DQN, self).__init__()

        picture_n, feature_n = state_n[0], state_n[1]
        picture_n = cv2.resize(picture_n, [250, 498])
        # input 4 x 250 x 498
        input_ch = np.shape(picture_n)[-1]
        self.conv1 = nn.Conv2d(input_ch, 32, kernel_size=3, stride=5).to(device)  # 32 x 50 x 100
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)        # 64 x 25 x 50
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=4).to(device)        # 64 x

    def forward(self, x):
        img, feature = x[0], x[1]
        img = cv2.resize(img, [250, 498])

        img = Func.relu(self.conv1(img))
        img = Func.relu(self.conv2(img))
        img = Func.relu(self.conv3(img))
        img.view()