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

# 1채널 흑백 이미지인 기존 MNIST 데이터를 3채널 RGB로 변환 [1, 28, 28] → [3, 28, 28]
def to_rgb(x):
    return x.repeat(3, 1, 1)

transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(to_rgb),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])
train_transform = transforms.Compose([
    # -15 ~ +15도 사이의 무작위 회전
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Lambda(to_rgb),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])
test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

data_dir = './data'
test_dir = 'data/mnist_test'

train_dataset = datasets.MNIST(root=data_dir,
                               train=True,
                               download=True,
                               transform=train_transform)
no_aug = datasets.MNIST(root=data_dir,
                               train=True,
                               download=True,
                               transform=transform_no_aug)

index = 8

# 증강 전 데이터 가져오기
image_no_aug, label_no_aug = no_aug[index]
# 증강 후 데이터 가져오기
image_with_aug, label_with_aug = train_dataset[index]
# 이미지 출력
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# 증강 전 이미지
image_no_aug = image_no_aug.permute(1,2,0).numpy().clip(0,1)
axes[0].imshow(image_no_aug)
axes[0].set_title(f'Original: {label_no_aug}')
axes[0].axis('off')
# 증강 후 이미지
image_with_aug = image_with_aug.permute(1,2,0).numpy().clip(0,1)
axes[1].imshow(image_with_aug)
axes[1].set_title(f'Augmented: {label_with_aug}')
axes[1].axis('off')
plt.show()