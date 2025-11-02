# coding: utf-8
import os.path
import re

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from PIL import Image


def dataProcessing(csvPath):
    data = pd.read_csv(csvPath, encoding='utf-8')
    urls = data['Image Url']
    texts = data['Title']
    labels = data['label']
    datas = []
    for i in range(len(urls)):
        datas.append([urls[i].split('/')[-2], texts[i], labels[i]])
    print('数据条数', len(data['Image Url']))
    return urls, datas


def download_img(urls):
    img_paths = []
    for img_url in tqdm(urls):
        header = {"Authorization": "Bearer " }  # 设置http header，视情况加需要的条目，这里的token是用来鉴权的一种方式
        r = requests.get(img_url, headers=header, stream=True)
        if r.status_code == 200:
            open('./data/wechat/' + img_url.split('/')[-2] +'.jpeg', 'wb').write(r.content)  # 将内容写入图片
        else:
            print(img_url.split('/')[-2] +'.jpeg')


if __name__ == '__main__':
    img_paths = pd.read_csv('../data/wechat/data.csv')['path']
    for i in img_paths:
        if not os.path.exists(i):
            print(i)

    # urls, _ = dataProcessing('./data/wechat/test/news.csv')
    # download_img(urls)
