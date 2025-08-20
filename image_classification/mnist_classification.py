import torch
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
# 역정규화 함수
def denormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

# 이미지 5개씩 출력 (역정규화 적용)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
plt.figure(figsize=(12, 6))
for i in range(5):
    img, label = train_dataset[i]
    print(f"train image.shape: {img.shape}")
    img_show = denormalize(img, mean, std)
    plt.subplot(2, 5, i+1)
    plt.imshow(img_show.permute(1, 2, 0))
    plt.title(f"Train: {label}")
    plt.axis('off')

for i in range(5):
    img, label = test_dataset[i]
    print(f"test image.shape: {img.shape}")
    img_show = denormalize(img, mean, std)
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(img_show.permute(1, 2, 0).clip(0, 1))
    plt.title(f"Test: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)