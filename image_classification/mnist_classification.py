import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
import glob
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        label_str = filename.split('_')[1]
        label = int(label_str)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.ToTensor(),
    # -15 ~ +15도 사이의 무작위 회전
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './data'
test_dir = 'data/mnist_test'

train_dataset = datasets.MNIST(root=data_dir,
                               train=True,
                               download=True,
                               transform=train_transform)
test_dataset = CustomDataset(test_dir,
                             transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# MNIST 분류를 위해 FC수정
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()