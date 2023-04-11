import pandas as pd
import os
import csv
import numpy as np
import requests
from tqdm import tqdm

# 原始数据重新整理
imgs = os.listdir('../data/twitter/images/')
root_path = '../data/'


# print(imgs)
def new_data(path, label):
    len_list = []
    dataList = []
    with open(path, 'r', encoding='utf-8') as t1:
        t1 = t1.readlines()
        if len(t1) % 3 == 0:
            num = int(len(t1) / 3)
            print('数据列数:', len(t1))
            print('数据条数(除以3):', num)
            for n in range(num):
                #  a2为图片名，a3为文本
                a1, a2, a3 = t1[n * 3], t1[n * 3 + 1], t1[n * 3 + 2]
                a2 = a2.strip()  # 取出换行符

                text_len = len(a3)
                len_list.append(text_len)

                a2 = a2.split('|')  # 分割图片
                a2 = [x.split('/')[-1] for x in a2 if x != 'null']  # 去除空数据并分割出图片路径
                a2 = [root_path + 'twitter/images/' + x for x in a2 if x in imgs]

                for i in range(len(a2)):
                    dataList.append([a2[i], a3.strip(), label])

            print('平均句子长度:', np.mean(len_list))
            return dataList

        else:
            print('数据长度不合理')


def list2CSV(dataList, path):
    # if os.path.exists(path):
    #     os.remove(path)  #
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('path', 'text', 'label'))

    for data in dataList:
        all_info = data[0], data[1], data[2]  # 路径， 文本， 标签（0，1）
        with open(path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(all_info)


def dataProcessing1():  # 处理twitter数据集
    csvTestPath = '../data/twitter/test.csv'
    csvTrainPath = '../data/twitter/train.csv'
    if os.path.exists(csvTestPath) and os.path.exists(csvTrainPath):
        os.remove(csvTestPath)  # 如果存在就删除以免重复写入
        os.remove(csvTrainPath)

    with open(csvTrainPath, 'a', encoding='utf-8', newline='') as f:  # 写入列名
        writer = csv.writer(f)
        writer.writerow(('path', 'text', 'label'))

    with open(csvTestPath, 'a', encoding='utf-8', newline='') as f:  # 写入列名
        writer = csv.writer(f)
        writer.writerow(('path', 'text', 'label'))

    list2CSV(new_data('../data/twitter/tweets/train_rumor.txt', 1), csvTrainPath)  # 训练谣言数据，1表示给谣言数据添加标签1
    list2CSV(new_data('../data/twitter/tweets/train_nonrumor.txt', 0), csvTrainPath)  # 训练非谣言数据，0表示给谣言数据添加标签0
    list2CSV(new_data('../data/twitter/tweets/test_rumor.txt', 1), csvTestPath)
    list2CSV(new_data('../data/twitter/tweets/test_nonrumor.txt', 0), csvTestPath)


def dataProcessing2(csvPath):  # 处理WeChat数据集
    data = pd.read_csv(csvPath, encoding='utf-8')
    urls = data['Image Url']
    texts = data['Title']
    labels = data['label']
    datas = []
    for i in range(len(urls)):
        datas.append(['../data/wechat/images/' + urls[i].split('/')[-2] + '.jpeg', texts[i], labels[i]])
    print('数据条数', len(data['Image Url']))
    return urls, datas


def download_img(urls):
    for img_url in tqdm(urls.tolist()):
        header = {"Authorization": "Bearer "}  # 设置http header，视情况加需要的条目，这里的token是用来鉴权的一种方式
        r = requests.get(img_url, headers=header, stream=True)
        if r.status_code == 200:
            img_path = img_url.split('/')[-2]
            open('../data/wechat/images/' + img_path + '.jpeg', 'wb').write(r.content)  # 将内容写入图片
        else:
            print(img_url.split('/')[-2] + '.jpeg')


def split_dataset(csv_path1, csv_path2, target_path):
    test_df = pd.read_csv(csv_path1, encoding='utf-8')
    train_df = pd.read_csv(csv_path2, encoding='utf-8')
    val_df = train_df.sample(frac=0.1)
    train_df = train_df.drop(index=val_df.index.to_list())

    print('训练集长度:', len(train_df))
    print('验证集长度:', len(val_df))
    print('测试集长度', len(test_df))

    test_df.to_csv(target_path + 'test.csv', encoding='utf-8', index=None)
    val_df.to_csv(target_path + 'val.csv', encoding='utf-8', index=None)
    train_df.to_csv(target_path + 'train.csv', encoding='utf-8', index=None)


if __name__ == '__main__':

    # dataProcessing1()
    # split_dataset('../data/twitter/test.csv', '../data/twitter/train.csv', '../data/twitter/')

    # urls, datas = dataProcessing2('../data/wechat/train/news.csv')
    # list2CSV(datas, '../data/wechat/train.csv')
    # urls, datas = dataProcessing2('../data/wechat/test/news.csv')
    # list2CSV(datas, '../data/wechat/test.csv')
    # split_dataset('../data/wechat/test.csv', '../data/wechat/train.csv', '../data/wechat/')

    datas = pd.read_csv('../data/wechat/test.csv')
    datas = datas.drop(datas.loc[datas["label"] == 0].sample(frac=0.8).index.to_list())
    print(len(datas))
    datas.to_csv('../data/wechat/test.csv',index=None)

    # change = {}
    # path = datas['path'].tolist()
    # for item in path:
    #     new_path = '../data/wechat/images/' + item + '.jpeg'
    #     change[item] = new_path
    # datas['path'] = datas['path'].map(change)
    # datas.to_csv('../data/wechat/newData.csv', index=False)



