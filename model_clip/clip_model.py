# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from tensorboardX import SummaryWriter

from utils import My_Dataset, get_time_dif
from Config import Config
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPModel, CLIPProcessor, BertForSequenceClassification


def train(config, model, train_iter, dev_iter, test_iter,writer):
    start_time = time.time()
    model.train()
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters()]}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config.other_learning_rate)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1, last_epoch=-1)  # 每2个epoch学习率衰减为原来的一半

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        loss_list = []  # 承接每个batch的loss
        acc_list = []
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            true = labels.data.cpu()
            predict = torch.max(outputs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predict)

            writer.add_scalar('train/loss_iter', loss.item(), total_batch)
            writer.add_scalar('train/acc_iter', train_acc, total_batch)
            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'

            if total_batch % 1000 == 0:
                print(msg1.format(total_batch, loss.item(), train_acc))

            loss_list.append(loss.item())
            acc_list.append(train_acc)

            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过2000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        # if flag:
        #     break

        dev_acc, dev_loss = evaluate(config, model, dev_iter)  # model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        epoch_loss = np.mean(loss_list)
        epoch_acc = np.mean(acc_list)
        msg2 = 'EPOCH: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
        print(msg2.format(epoch+1, epoch_loss, epoch_acc, dev_loss, dev_acc, time_dif, improve))
        writer.add_scalar('train/loss_epoch', epoch_loss, epoch)
        writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
        writer.add_scalar('val/loss_epoch', dev_loss, epoch)
        writer.add_scalar('val/acc_epoch', dev_acc, epoch)

        model.train()
        scheduler.step()
        print('epoch: ', epoch, 'lr: ', scheduler.get_last_lr())

    test(config, model, test_iter)


def test(config, model, test_iter):
    # 测试函数
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...") #精确率和召回率以及调和平均数
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()  ###预测结果
            # print(outputs)
            # print(predic)
            # print(labels)
            # print('*************************')
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


class CLIPNet(nn.Module):
    def __init__(self, config):
        super(CLIPNet, self).__init__()
        self.config = config
        self.text_encoder = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.fc_1 = nn.Linear(self.config.bert_fc + self.config.resnet_fc, self.config.num_classes)
        self.drop = nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inx):
        img, tokens = inx
        img = self.clip_model.get_image_features(pixel_values=img)
        outputs = self.text_encoder(tokens).logits
        result = torch.cat([img, outputs], 1)
        result = self.fc_1(result)
        result = self.softmax(result)
        return result


if __name__ == '__main__':
    config = Config()
    writer = SummaryWriter(log_dir=config.log_dir)

    # Dataloader 输出 image vector + text token vector
    train_data = My_Dataset('../data/twitter/train.csv', config, 1)
    dev_data = My_Dataset('../data/twitter/val.csv', config, 1)
    test_data = My_Dataset('../data/twitter/test.csv', config, 1)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)   # 训练迭代器
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True)      # 验证迭代器
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)   # 测试迭代器

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    print("Loading data...")

    # 训练
    CLIP_model = CLIPNet(config)
    # 模型放入到GPU中去
    CLIP_model = CLIP_model.to(config.device)
    print(CLIP_model.parameters)

    # #训练结束后可以注释掉train函数只跑test评估模型性能
    # #test(config, CLIP_model, test_iter)
    train(config, CLIP_model, train_iter, dev_iter, test_iter, writer)
