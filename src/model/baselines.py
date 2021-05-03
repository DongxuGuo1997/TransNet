import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from .basenet import *


class Res18Crop(torch.nn.Module):
    """
    CNN for single frame model with cropped image.
    """

    def __init__(self, backbone, drop_p=0.5):
        super(Res18Crop, self).__init__()
        self.backbone = backbone
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_p)
        self.act = nn.Sigmoid()
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(-1, 1024)
        x = self.linear(x)
        x = self.act(x)

        return x


class Res18RoI(torch.nn.Module):
    """
    CNN for single frame model. ResNet-18 bacbone and RoI for adding context.
    """

    def __init__(self, resnet, last_block, drop_p=0.5):
        super(Res18RoI, self).__init__()
        self.resnet = resnet
        self.last_block = last_block.apply(set_conv2d_stride1)
        self.conv_last = torch.nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_p)
        self.FC = torch.nn.Linear(1024, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, imgs, bboxes):
        feature_maps = self.resnet(imgs)
        fa = RoIAlign(output_size=(7, 7), spatial_scale=1 / 8,
                      sampling_ratio=2, aligned=True)
        ya = fa(feature_maps, bboxes)
        y = self.last_block(ya)
        y = self.conv_last(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y.view(1, -1)
        y = self.FC(y)
        y = self.act(y)

        return y


class Res18CropEncoder(nn.Module):
    def __init__(self, resnet):
        super(Res18CropEncoder, self).__init__()

        self.resnet = resnet
        # self.fc = nn.Linear(1024, CNN_embed_dim)

    def forward(self, x_5d, x_lengths):
        x_seq = []
        for i in range(x_5d.size(0)):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                with torch.no_grad():
                    img = x_5d[i, t, :, :, :]
                    x = self.resnet(torch.unsqueeze(img, dim=0))  # ResNet
                    x = x.view(x.size(0), -1)  # flatten output of conv
                cnn_embed_seq.append(x)
                # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1, fea_dim)
            x_seq.append(embed_seq)
        # pad feature vector sequences
        x_padded = nn.utils.rnn.pad_sequence(x_seq, batch_first=True, padding_value=0)

        return x_padded


class Res18RoIEncoder(nn.Module):
    def __init__(self, encoder):
        super(Res18RoIEncoder, self).__init__()

        self.encoder = encoder
        # self.fc = nn.Linear(1024, CNN_embed_dim)

    def forward(self, x_5d, bbox_list, x_lengths):
        x_seq = []
        for i in range(x_5d.size(0)):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                with torch.no_grad():
                    img = x_5d[i, t, :, :, :]
                    bbox = [bbox_list[t][i]]
                    x = self.encoder(torch.unsqueeze(img, dim=0), bbox)  # ResNet
                cnn_embed_seq.append(x)
                # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1, fea_dim)
            x_seq.append(embed_seq)

        x_padded = nn.utils.rnn.pad_sequence(x_seq, batch_first=True, padding_value=0)

        return x_padded


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim=1024, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.2):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.dropout = nn.Dropout(p=drop_p)
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x_3d, x_lengths):
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_3d, x_lengths, batch_first=True, enforce_sorted=False)
        self.LSTM.flatten_parameters()
        packed_RNN_out, _ = self.LSTM(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)

        return x
