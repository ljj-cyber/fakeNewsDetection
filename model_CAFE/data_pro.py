'''我们吧数据处理成跟原文差不多的形式,处理好的数据保存在mydata文件夹下'''

import numpy as np
from transformers import BertTokenizer,BertModel
import torch
from torchvision.models import resnet34,resnet50
from PIL import Image
import cv2
import pandas as pd
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

# text=np.load('org_data/train_text_with_label.npz')
# img=np.load('org_data/train_image_with_label.npz')
# print(text['data'].shape)
# print(text['label'][:10])
# print(img['data'].shape)
# print(img['label'][:10])
tokenizer = BertTokenizer.from_pretrained('../bert_model/minirbt-h256')
model = BertModel.from_pretrained('../bert_model/minirbt-h256')
model.to(device)
model.eval()

resnet = resnet34(pretrained=True)
# print(resnet)
resnet.fc = torch.nn.Linear(512, 512)
resnet.to(device)
resnet.eval()
def encode_text(text):
    try:
        len(text)
    except:
        text=''
    text = tokenizer(text=text, add_special_tokens=True,
                          max_length=128,  # 最大句子长度
                          padding='max_length',  # 补零到最大长度
                          truncation=True,
                     return_tensors='pt')
    input_id = text['input_ids']
    input_id = input_id.to(device)
    out = model(input_ids=input_id)
    res = out[0]
    res = res.squeeze(0)
    res = res.data.cpu().numpy()


    return res

def encode_img(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))  #
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    img = img.to(device)

    res = resnet(img)
    res = res.data.cpu().numpy()

    res = res.reshape(512, 1, 1)
    return res


def pro_data(path, howstyle, dataset):
    df = pd.read_csv(path)
    arr_img = []
    arr_text = []
    arr_label = []
    for x in tqdm(range(len(df))):
        path, text, label = df.iloc[x, :]
        eimg = encode_img(path)
        etext = encode_text(text)
        arr_img.append(eimg)
        arr_text.append(etext)
        arr_label.append(label)
    arr_img = np.array(arr_img)
    arr_text = np.array(arr_text)
    arr_label = np.array(arr_label)
    print(arr_img.shape)
    print(arr_text.shape)
    np.savez('mydata/{}_{}_img.npz'.format(howstyle, dataset), data=arr_img, label=arr_label)
    np.savez('mydata/{}_{}_text.npz'.format(howstyle, dataset), data=arr_text, label=arr_label)


pro_data('../data/twitter/train.csv', 'train', 'twitter')
pro_data('../data/twitter/val.csv', 'val', 'twitter')
pro_data('../data/twitter/test.csv', 'test', 'twitter')