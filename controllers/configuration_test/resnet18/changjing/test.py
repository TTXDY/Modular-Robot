import os

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import DogCatDataset


def main():
    # Step 0:查看torch版本、设置device
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1:准备数据集
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_data = DogCatDataset.DogCatDataset(root_path=os.path.join(os.getcwd(), 'data/test'),
                                            transform=test_transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # Step 2: 初始化网络
    model = models.resnet18()

    # 修改网络结构，将fc层1000个输出改为2个输出
    fc_input_feature = model.fc.in_features
    model.fc = nn.Linear(fc_input_feature, 2)

    # Step 3：加载训练好的权重
    trained_weight = torch.load('./resnet18_Step_Flat4.pth')
    model.load_state_dict(trained_weight)
    model.to(device)

    # Steo 4：网络推理
    model.eval()

    correct_sample = 0
    total_sample = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            # print(type(img))
            img = img.to(device)
            label = label.to(device)
            output = model(img)

            _, predicted_label = torch.max(output, 1)

            correct_sample += (predicted_label == label).cpu().numpy()
            total_sample += 1

    # Step 5:打印分类准确率
    print(correct_sample / total_sample)


if __name__ == '__main__':
    main()
