## 概述

这个工程主要用来给计算机视觉小白练手，基于Pytorch深度学习框架实现。

使用猫狗分类数据集中的训练集，共25000张图片。将原始训练集进行拆分，其中20000张用于训练，其余5000张用于测试。

分类网络使用ResNet-18，使用了交叉熵损失函数和SGD优化方法。


## 环境配置

在Ubuntu16.04系统下，建立Conda虚拟环境，python3.7，几个重要的库：

（1）pytorch     1.7.0

（2）torchvision 0.8.0

（3）opencv-python  4.5.2.52

（4）tqdm    4.61.0

## 目录结构
```
.
├── data
│   ├── newtest
│   ├── newtrain
│   ├── train
│   └── train.zip
├── DogCatDataset.py
├── prepare_data.py
├── README.md
├── resnet18_Cat_Dog.pth
├── test.py
└── train.py
```

## 数据集下载

百度网盘链接：https://pan.baidu.com/s/19eG-kbPifVfIRGcgS21gZA 

提取码：hjag 

## 训练好的权重下载

百度网盘链接：https://pan.baidu.com/s/1DykBh0ht5URLzdludSVPNQ 

提取码：s7h2 

## 运行方法
必须下载数据集。若要自己训练，不需要下载训练好的权重；若想直接测试，需要下载训练好的权重。

数据集下载完成后，存放在```[工程主目录]/data```路径下，首先运行如下命令完成数据集划分
```python
python prepare_data.py
```
运行完成后，```[工程主目录]/data```路径下会生成```newtrain```和```newtest```这2个路径，分别存放训练集和测试集。

**训练**
```python
python train.py
```
训练完成后，在工程主目录下会生成名为```resnet18_Cat_Dog.pth```的权重文件，推理时会读取该权重文件。

**推理**
```python
python test.py
```
推理完成后会打印出推理的正确率。

## 写在最后

大家在运行过程中有什么疑问，欢迎在Gitee上提issue，也可以在公众号留言。

如果你对计算机视觉中的目标检测、跟踪、分割、轻量化神经网络感兴趣，欢迎扫描下方二维码关注公众号一起学习交流

<div align=center> 
<img src="./CV365.jpg"/>
</div>