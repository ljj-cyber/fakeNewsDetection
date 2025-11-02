import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from transformers import BertTokenizer, CLIPProcessor
from Config import Config
import clip


class My_Dataset(Dataset):
    def __init__(self, path, config, iftrain):  # 读取数据集
        self.config = config
        # 启用训练模式，加载数据和标签
        self.iftrain = iftrain
        df = pd.read_csv(path).sample(frac=self.config.frac)
        self.img_path = df['path'].to_list()
        self.text = df['text'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # 启用训练模式，加载数据和标签
        if self.iftrain == 1:
            self.labels = df['label'].to_list()

    def __getitem__(self, idx):
        img = self.processor(images=Image.open(self.img_path[idx]))
        img = torch.from_numpy(np.array(img['pixel_values'][0]))
        text = self.text[idx]
        try:
            len(text)  # 部分文本是nan
        except:
            text = ''
        tokens = self.tokenizer(text=text, add_special_tokens=True,
                                max_length=self.config.pad_size,  # 最大句子长度
                                padding='max_length',  # 补零到最大长度
                                truncation=True)['input_ids']
        tokens = torch.tensor(tokens, dtype=torch.long)

        if self.iftrain == 1:
            label = int(self.labels[idx])
            label = torch.tensor(label, dtype=torch.long)
            return ((img.to(self.config.device), tokens.to(self.config.device)), label.to(self.config.device))

        else:
            return (img.to(self.config.device), tokens.to(self.config.device))

    def __len__(self):
        return len(self.img_path)  # 总数据长度


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config = Config()
    train_data = My_Dataset('../data/twitter/train.csv', config, 1)
    train_iter = DataLoader(train_data, batch_size=32)

    for n, (a, b) in enumerate(train_iter):
        print(n, a[0].shape)
        #print(y)
        print('************')
