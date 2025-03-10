import dataLoad
from torch.utils.data import TensorDataset, DataLoader
import argparse
import torch
import network
import torch.nn as nn
from myDataSet import CustomDataset
import numpy as np
import utils
import time
from visualize import tsne_and_save,tsne_and_save1
from centerLoss import CenterLoss
import math
import torch.optim as optim
from evaluation import Evaluation
import matplotlib.pyplot as plt

#add decoder reconstruct

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--lr-cent', type=float, default=0.0001, help="learning rate for center loss")
parser.add_argument('--hid', type=int, default=256, help = 'hidden layer size')
parser.add_argument('--kernel_size', type=int, default=100, help = "kernel size")
parser.add_argument('--step_size', type=int, default=2, help = "step_size")
parser.add_argument('--batch_size', type=int, default=100, help = 'batch size')
parser.add_argument('--lr', type=float, default=0.0001, help = 'learning rate')
parser.add_argument('--K', type=int, default=2, help = 'subdomain number')
parser.add_argument('--epochs', type=int, default=60, help = 'number of epochs')

args = parser.parse_args()
# train,test,label = dataLoad.loadSwat()
train,test,label = dataLoad.loaKdd()
train = train.astype(float)
test = test.astype(float)
label = label.astype(int)
n,t = train.shape
tn,tt = test.shape


trainSeg = np.zeros((n, math.floor((t-args.kernel_size)/args.step_size + 1), args.kernel_size))
testSeg = np.zeros((n, math.floor((tt-args.kernel_size)/args.step_size + 1), args.kernel_size))
testlabel = np.zeros(math.floor((tt-args.kernel_size)/args.step_size + 1))
trainlabel = np.zeros(math.floor((t-args.kernel_size)/args.step_size + 1))

for i in range(0,t-args.kernel_size+1,args.step_size):
    index = i //  args.step_size
    trainSeg[:, index, :] = train[:, i:i+args.kernel_size]

for i in range(0,tt-args.kernel_size+1,args.step_size):
    index = i // args.step_size
    testSeg[:, index, :] = test[:, i:i+args.kernel_size]
    testlabel[index] = label[i+args.kernel_size-1]

print(len(testlabel))
print(sum(testlabel==1))

train_dLabel = torch.zeros(math.floor((t-args.kernel_size)/args.step_size + 1)).type(torch.LongTensor).to(device)
test_dLabel = torch.zeros(math.floor((tt-args.kernel_size)/args.step_size + 1)).type(torch.LongTensor).to(device)

testlabel = torch.tensor(testlabel).type(torch.LongTensor).to(device)

dataset = CustomDataset(torch.tensor(trainSeg).permute(1,0,2).type(torch.float32).to(device),
                        torch.tensor(trainlabel).type(torch.LongTensor).to(device))

testset = CustomDataset(torch.tensor(testSeg).permute(1,0,2).type(torch.float32).to(device),testlabel
                        )
dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
testLoader = DataLoader(testset, args.batch_size, shuffle=True)


criterion_cent = CenterLoss(num_classes=args.K, feat_dim=256, use_gpu=True)
optimizer_centloss = torch.optim.Adam(criterion_cent.parameters(), lr=args.lr_cent)
domainCross = nn.CrossEntropyLoss()
centerLoss = nn.MSELoss()
reconstructLoss = nn.MSELoss()
classCross = nn.CrossEntropyLoss()
lambda_center = 1
lambda_rec = 1

def train_step(loader,testloader,epoch,val=False):
    domain1= []
    domain2 = []
    rec =[]
    cent =[]
    threshold = 0.1
    best = 0.0
    if not val:
        for j, data in enumerate(loader):
            trainData,train_label = data
            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
            feature1, cls_feature1, new1,classify1 = model(trainData)
            feature2, cls_feature2, new2,_ = model(testset.trainData)
            b, d = cls_feature1.shape
            # 计算子域中心和对应伪标签 初始域中心都为0
            with torch.no_grad():
                mu1, d_label1 = utils.get_featureMu(feature1, torch.softmax(cls_feature1, dim=-1), args.K,device)
                mu2, d_label2 = utils.get_featureMu(feature2, torch.softmax(cls_feature2, dim=-1), args.K,device)
                train_dLabel = d_label1
                test_dLabel = d_label2
            domainLoss1 = domainCross(cls_feature1, train_dLabel)
            domainLoss2 = domainCross(cls_feature2, test_dLabel)
            domain1.append(domainLoss1.item())
            domain2.append(domainLoss2.item())
            new1 = new1.view(b, -1, args.kernel_size)
            new2 = new2.view(len(testset.trainData), -1, args.kernel_size)
            recLoss = reconstructLoss(trainData,new1) + reconstructLoss(testset.trainData,new2)
            rec.append(recLoss.item())
            center,common_center = criterion_cent(torch.cat((feature1,feature2),dim=0),
                                    torch.cat((d_label1,
                                               d_label2)))
            cent.append(center.item())
            class_loss = classCross(classify1,train_label)
            loss =1* (domainLoss1+ 2 * domainLoss2) + lambda_rec * recLoss + lambda_center * center + class_loss
            loss.backward()
            optimizer.step()
            optimizer_centloss.step()
    else:
        feature, cls_feature, _,ad_result = model(testset.trainData)
        mu1, d_label1 = utils.get_featureMu(feature, torch.softmax(cls_feature, dim=-1), args.K, device)
        # evaluation
        if epoch % 20 == 0 and epoch != 0:
            distance = utils.get_label(feature, torch.softmax(cls_feature, dim=-1), mu1, args.K,
                                       device)
            result_vector = torch.where(distance < threshold.to(device),
                                        torch.tensor(0).to(device), torch.tensor(1).to(device))
            eval_softmax = Evaluation(result_vector.detach().cpu().numpy(), testlabel.detach().cpu().numpy())
            if eval_softmax.f1_measure > best:
                best = eval_softmax.f1_measure
                pre = eval_softmax.precision
                reca = eval_softmax.recall
model = network.encoder(feature_dim=args.kernel_size * n, kernel_size=args.kernel_size,
                                hidden=args.hid, domainNum=args.K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best = 0.0
pre = 0.0
reca = 0.0
best1 = 0.0
pre1 = 0.0
reca1 = 0.0
for epoch in range(args.epochs):
    temp = None
    model.train()
    train_step(dataloader, testLoader, epoch)
    model.eval()
    with torch.no_grad():
        train_step(dataloader, testLoader, epoch, True)
print(f"Softmax F1 is %.6f" % (best))
print(f"Softmax precision is %.6f" % (pre))
print(f"Softmax recall is %.6f" % (reca))

