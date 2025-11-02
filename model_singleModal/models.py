import torch.nn as nn
import torch
from resnet_models import *


class Mynet(nn.Module):
    def __init__(self, config):
        super(Mynet, self).__init__()
        self.config = config
        resnet_name = self.config.resnet_name

        if config.modal == 'visual':
            if resnet_name == 'resnet18':
                self.resnet = resnet18(self.config.resnet_fc)
            elif resnet_name == 'resnet34':
                self.resnet = resnet34(self.config.resnet_fc)
            elif resnet_name == 'resnet50':
                self.resnet = resnet50(self.config.resnet_fc)
            elif resnet_name == 'resnet101':
                self.resnet = resnet101(self.config.resnet_fc)
            elif resnet_name == 'resnet152':
                self.resnet = resnet152(self.config.resnet_fc)
        elif config.modal == 'text':
            self.bert = BertModel.from_pretrained(self.config.bert_name)

        self.fc_1 = nn.Linear(self.config.bert_fc, self.config.num_classes)
        self.drop = nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inx):
        img, tokens, mask = inx

        if self.config.modal == 'visual':
            result = self.resnet(img)
        elif self.config.modal == 'text':
            result = self.bert(tokens)
            result = result[1]
        result = self.drop(result)

        result = self.fc_1(result)
        result = self.softmax(result)
        return result
