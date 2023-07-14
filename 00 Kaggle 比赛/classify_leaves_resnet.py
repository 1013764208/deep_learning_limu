import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # transforms 用于对图像进行各种变换
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm   # 可视化代码执行的进度


# 看下文件
labels_dataframe = pd.read_csv('../data/classify-leaves/train.csv')
print(labels_dataframe.head(5))
# print(labels_dataframe.describe())


# 1.数据预处理
# 将数据按标签列去重并排序
# set() 函数创建一个无序不重复元素集
# list() 创建列表
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# print(n_classes)  # 176 
# print(leaves_labels[:10]) # ['abies_concolor', 'abies_nordmanniana', 'acer_campestre'..


# 把label转成对应的数字
"""
    将标签转换对应的数字的作用：
    1.通常算法要求输入的标签是数字形式而非文本形式
    2.便于计算和索引，使用索引更高效
"""
class_to_num = dict(zip(leaves_labels, range(n_classes)))
# print(class_to_num)  # {'abies_concolor': 0, 'abies_nordmanniana': 1, 'acer_campestre': 2, 'acer_ginnala': 3 ..


# 再转换回来，方便最后预测的时候使用
num_to_class = {v : k for k, v in class_to_num.items()}
# print(num_to_class)   # {0: 'abies_concolor', 1: 'abies_nordmanniana'..


# 继承dataset，创建自己的
class LeavseData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path(string): csv 文件路径
            img_path(string): 图像文件所在路径
            mode(string): 训练模型还是测试模型
            valid_ratio(float): 验证集比例
        """
        # 需要调整后的照片尺寸
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode
        
        
        # 读取文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None 去掉表头
        # 计算lenght(count)
        self.data_len = len(self.data_info.index) -1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列是图像的名称
            # iloc[a,b] 索引第a行第b列的数据 [1:self_train_len,0] 多行
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])  # self.data_info.iloc[1:0] 表示读取第一列，从第二行到train_len
            # 第二列是图像的label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1]) 
            self.image_arr = self.train_image
            self.label_arr = self.train_label

        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label

        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finshed reading the {} set of Leaves Dataset ({} sample found)'.format(mode, self.real_len))

    
    def __getitem__(self, index):
        """
            用于在对象中实现索引操作
        """
        # 从image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 设置好需要转换的变量，还可以包括一些列的nomarlize等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转 选择一个概率（图像增强）
                transforms.RandomVerticalFlip(p=0.5),     # 随机水平反转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])


        # 将图像数据转换为张量
        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            # 返回每一个 index 对应的图像数据和对应的label
            return img_as_img, number_label  
        
    
    def __len__(self):
        return self.real_len
    

train_path = '../data/classify-leaves/train.csv'
test_path = '../data/classify-leaves/test.csv'
img_path = '../data/classify-leaves/'


# 读取 训练集，验证集，测试集
train_dataset = LeavseData(train_path, img_path, mode='train')
val_dataset = LeavseData(train_path, img_path, mode='valid')
test_dataset = LeavseData(test_path, img_path, mode='test')



# 定义数据加载器，可以按一定规则提取数据，方便多线程，批次和shuffle
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16, 
        shuffle=False,
        num_workers=5
    )

val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=16, 
        shuffle=False,
        num_workers=5
    )

test_loader = DataLoader(
        dataset=test_dataset, # 数据集
        batch_size=16,        # 每批加载数量
        shuffle=False,        # 是否打乱数据
        num_workers=5         # 子进程加载数据的数量
    )


# 看下是在cpu还是gpu上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)



# 2.模型
# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requiers_gred = False

# resnet34模型
def res_model(num_classes, features_extract=False, use_petrained=True):

    model_ft = models.resnet34(pretrained=use_petrained)
    set_parameter_requires_grad(model_ft, features_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )

    return model_ft


# resnet50模型
"""
    构造 resent 模型：
    num_classes 预训练模型的使用情况
    features_extract 特征提取器的使用情况
    use_pretrained 是否预训练权重
    return 返回模型对象
"""
def resnet_model(num_classes, features_extract=False, use_pretrained=True):
    
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    # 将提取模型的最后一层的输入特征数
    num_ftrs = model_ft.fc.in_features
    # 将创建一个新的线性层，该层将替换模型的最后一层，并将其输出大小设置为 num_classes
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft



# 3.设置超参数
learning_rate = 3e-4  # 学习率
weight_decay = 1e-3   # 权重衰退，正则化计算，用来限制模型的参数大小，以减少过拟合的风险
num_epoch = 50        # 训练次数


# model_path = './model/pre_resnext_model.ckpt'  # 预训练权重参数，迁移学习
# 设置保存预测结果的文件
saveFileName = './submission.csv'

# 创建模型 restnet
model = resnet_model(176)

# 将模型移动到指定的设备上进行加速计算
model = model.to(device)

# 模型加载预训练参数
# model.load_state_dict(torch.load(model_path))   # if no load weights from checkpoint train

# 定义损失函数，这里使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 初始化优化器，优化器用于根据计算得到的梯度来更新模型的参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 控制训练的轮数
n_epochs = num_epoch

# 初始化变量
best_acc = 0.0


# 4.训练
for epoch in range(n_epochs):  
    # 将模型设置为训练模型
    model.train()

    # 用于记录每个批次的训练损失和准确率
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        # 将批次中的图像数据和对应的标签分离出来
        imgs, labels = batch

        # 将数据移动到指定的设备上进行加速计算
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 将图像数据输入模型，得到模型的结果
        logits = model(imgs)

        # 计算模型预测结果与真实标签间的损失
        loss = criterion(logits, labels)

        # 清空之前计算的梯度信息，防止梯度累积
        optimizer.zero_grad()

        # 根据损害函数的值，反向传播算法，计算模型参数的梯度
        loss.backward()

        # 使用优化器 根据计算得到的梯度来更新参数
        optimizer.step()

        # 计算当前批次的准确率
        acc = (logits.argmax(dim=-1) == labels).float().mean()  # argmax 找出概率最高的类型，与真实标签比较

        # 将当前批次的损失和准确率记录下来
        train_loss.append(loss.item())
        train_accs.append(acc)

    
    # 计算整个训练集上的平均损失和准确率
    train_loss = sum(train_loss) / len(train_loss)
    train_acc =  sum(train_accs) / len(train_accs)

    # print the information
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}")



# 5. 进行模型的验证并生成预测结果
# 将模型设置为评估模型，在评估模型下，模型不会进行参数更新
model.eval()  

# 初始化列表，用于存储预测结果
predictions = []

# iterate the testing set by batches
for batch in tqdm(test_loader):
    # 将批次中的图像数据提取出来
    imgs = batch

    # 使用上下问管理器，确保在验证阶段不进行梯度算计，以减少内存消耗和加速计算
    with torch.no_grad():
        # 将图像数据输入模型，得到模型结果
        logits = model(imgs.to(device)) 
    
    # 将每个批次的预测结果中概率最高的类型索引添加到 predic 列表中
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist()) 

preds = []
# 将预测结果数字转换为类型标签
for i in predictions:
    preds.append(num_to_class[i])

# 读取预测数据集文件
test_data = pd.read_csv(test_path)

# 将转换后的预测结果（类别标签）添加到测试数据集的'label'列
test_data['label'] = pd.Series(preds)

# 将测试数据集中的图像文件名和预测结果拼接一起，形成最终的提交结果
submission = pd.concat([test_data['image'], test_data['label']], axis=1)

# 将其结果转换为CSV文件 
submission.to_csv(saveFileName, index=False)
print("Done!")
