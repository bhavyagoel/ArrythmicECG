import torch
import torch.nn as nn


class InitConv(nn.Module):
    def __init__(self, input_size, dropout, shape):  # 0  for linear, 1 for square
        super(InitConv, self).__init__()
        self.shape = shape
        self.relu = nn.ReLU()
        self.dropout_ = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        if self.shape == 0:

            # Initial Convolution Model
            self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=2, bias=True)
            self.batch_norm_1d_1 = nn.BatchNorm1d(num_features=4, eps=0.00001, momentum=0.1)

            self.conv1d_2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=4, stride=2, bias=True)
            self.batch_norm_1d_2 = nn.BatchNorm1d(num_features=16, eps=0.00001, momentum=0.1)

            self.conv1d_3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=True)
            self.batch_norm_1d_3 = nn.BatchNorm1d(num_features=32, eps=0.00001, momentum=0.1)

            self.conv1d_4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, bias=True)
            self.batch_norm_1d_4 = nn.BatchNorm1d(num_features=32, eps=0.00001, momentum=0.1)

            self.linear_1d_1 = nn.Linear(in_features=7136, out_features=2048)
            self.linear_1d_2 = nn.Linear(in_features=2048, out_features=1024)
            self.linear_1d_3 = nn.Linear(in_features=1024, out_features=512)
            self.linear_1d_4 = nn.Linear(in_features=512, out_features=256)

            # Attention Model
            self.attention_linear_1d_1 = nn.Linear(in_features=256, out_features=256, bias=True)

        elif self.shape == 1:
            self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 2), stride=(2, 2), bias=True)
            self.batch_norm_2d_1 = nn.BatchNorm2d(num_features=4, eps=0.00001, momentum=0.1)

            self.conv2d_2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(2, 2), stride=(1, 2), padding=(2, 2),
                                      padding_mode="reflect")
            self.batch_norm_2d_2 = nn.BatchNorm2d(num_features=16, eps=0.00001, momentum=0.1)

            self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(2, 1),
                                      padding=(2, 2), padding_mode="reflect")
            self.batch_norm_2d_3 = nn.BatchNorm2d(num_features=32, eps=0.00001, momentum=0.1)

    def forward(self, inputs):
        if self.shape == 0:
            t = inputs
            t = self.conv1d_1(t)
            t = self.batch_norm_1d_1(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.conv1d_2(t)
            t = self.batch_norm_1d_2(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.conv1d_3(t)
            t = self.batch_norm_1d_3(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.conv1d_4(t)
            t = self.batch_norm_1d_4(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.flatten(t)

            t = self.linear_1d_1(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.linear_1d_2(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.linear_1d_3(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.linear_1d_4(t)

            t = self.dropout_(t)

            alpha = t.detach().clone()

            alpha = self.tanh(alpha)
            alpha = alpha.view(alpha.shape[0], 1, alpha.shape[1])

            beta = self.attention_linear_1d_1(alpha)
            beta = self.softmax(beta)
            gamma = beta.detach().clone()

            for _ in range(16):
                beta = self.attention_linear_1d_1(alpha)
                gamma = torch.cat((gamma, beta), dim=1)
            gamma = self.softmax(gamma)

            for _ in range(17):
                t = t + gamma[:, _, :]

            t = self.relu(t)
            return t

        elif self.shape == 1:
            t = inputs
            t = self.conv2d_1(t)
            t = self.batch_norm_2d_1(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.conv2d_2(t)
            t = self.batch_norm_2d_2(t)
            t = self.relu(t)
            t = self.dropout_(t)

            t = self.conv2d_3(t)
            t = self.batch_norm_2d_3(t)
            t = self.relu(t)

            t = self.flatten(t)

            return t


class CnnBiLSTM1D(nn.Module):
    def __init__(self, dropout):
        super(CnnBiLSTM1D, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout_ = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        # InitConv BLocks
        self.block_1 = InitConv(3600, 0.005, 0)
        self.block_2 = InitConv(3600, 0.005, 0)

        # Bi-Linear
        self.bilinear = nn.Bilinear(256, 256, 256)

        # Linear Layer
        self.linear_1 = nn.Linear(1024, 512, bias=True)
        self.linear_2 = nn.Linear(512, 256, bias=True)
        self.linear_3 = nn.Linear(256, 128, bias=True)
        self.linear_4 = nn.Linear(128, 64, bias=True)
        self.out = nn.Linear(64, 17, bias=True)

        # Conv Layer
        self.conv_1d_1 = nn.Conv1d(1, 8, kernel_size=4, stride=2, bias=True)

        # Batch Norm 1D
        self.batch_norm_1d_1 = nn.BatchNorm1d(8, eps=0.00001, momentum=0.1)

        # Bi-LSTM
        self.bi_lstm_1 = nn.LSTM(256, 512, num_layers=2, bias=True, dropout=0.005, bidirectional=True)
        self.bi_lstm_2 = nn.LSTM(1024, 1024, num_layers=2, bias=True, dropout=0.002, bidirectional=True)

    def forward(self, inputs):
        a = inputs.detach().clone()
        b = inputs.detach().clone()
        b = torch.flip(b, dims=[2])
        a = self.block_1(a)
        b = self.block_2(b)

        t = self.bilinear(a, b)

        t = t.view(t.shape[0], 1, t.shape[1])

        t, _ = self.bi_lstm_1(t)
        t = self.tanh(t)

        #         t, _ = self.bi_lstm_2(t)
        #         t = self.tanh(t)

        t = t.view(-1, t.shape[2])

        t = self.linear_1(t)
        t = self.relu(t)

        t = self.linear_2(t)
        t = self.relu(t)

        t = self.linear_3(t)
        t = self.relu(t)

        t = self.linear_4(t)
        t = self.relu(t)

        t = self.out(t)

        return t
