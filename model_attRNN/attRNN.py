import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_attRNN.Config import Config
from model_attRNN.utils import My_Dataset

import torchvision.models as models


class attRNN(nn.Module):
    def __init__(self, config):
        super(attRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=21128, embedding_dim=config.bert_fc)  # 21128 stands for the vocab size.
        self.lstm = nn.LSTM(input_size=config.bert_fc, hidden_size=512, num_layers=2, batch_first=True)
        self.att_fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.att_fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.Softmax()
        )
        self.vgg = models.vgg11(pretrained=True)
        self.vis_fc1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU()
        )
        self.vis_fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.Softmax()
        )
        self.all_fc = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Softmax()
        )

    def forward(self, data):
        image, tokens, _ = data

        text_vec = self.embedding(tokens)
        out, (hidden_state, cell_memory) = self.lstm(text_vec)
        hidden_state = torch.mean(hidden_state, dim=0)
        att_1 = self.att_fc1(hidden_state)
        att_weight = self.att_fc2(att_1)

        img = self.vgg(image)
        img_hid = self.vis_fc1(img)
        img_hid2 = self.vis_fc2(img_hid)

        vis = torch.mul(att_weight, img_hid2)
        all_fea = torch.cat([hidden_state, vis], dim=1)
        result = self.all_fc(all_fea)
        return result





