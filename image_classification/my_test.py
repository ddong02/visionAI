import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 0 if 'cat' in img_name else 1

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
transform = transform.Compose([
    transform.Resize(256),
    transform.CenterCrop(224),
    transform.ToTensor(),
    transform.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_data_path = 'data/test2'
test_dataset = TestDataset(test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

saved_model_path = 'data/output_tensorboard/final_model.pth'
model = torch.load(saved_model_path, weights_only=False)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"model loaded from '{saved_model_path}'")
print(f'using device: {device}')

# cat → 0, dog → 1
class_label = ['cat', 'dog']
# mps 환경에서 모델을 훈련했기 때문에 테스트 데이터셋과 모델을 동일한 장치에 위치하도록 해야함

# all_images = []
# all_labels = []
# # 리스트에 이미지 텐서와 레이블 텐서를 추가
# for image, label in test_loader:
#     all_images.extend(image)
#     all_labels.extend(label)

# fig = plt.figure(figsize=(20, 8))
# for j in range(len(all_images)):
#     ax = fig.add_subplot(2, 10, j+1)
#     # 텐서형태의 이미지를 넘파이 형태로 바꾸고 matplotlib에 맞게 차원을 변경해줌(H,W,C)
#     img = all_images[j].numpy().transpose((1, 2, 0))
#     # 모델 학습을 위해 정규화 된 이미지를 다시 역정규화 시킴. 정규화 → (img - mean) / std
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
#     plt.axis('off')
#     ax.imshow(img)
#     ax.set_title(f'{class_label[all_labels[j]]}')
# plt.show()

# 올바른 정답 횟수와 총 예측 횟수를 저장할 변수
correct_predictions = 0
total_predictions = 0

import time
# 검증을 위해 torch.no_grad()를 사용 → 기울기 계산이 필요하지 않기 때문에
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        # 데이터를 훈련 때와 같은 device로 이동시켜줘야 함.
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        total_predictions += labels.size(0)
        correct_predictions += (preds==labels).sum().item()
end_time = time.time()

accuracy = 100 * correct_predictions / total_predictions
print(f'Test Accuracy: {accuracy:.2f}%')
print(f"Test time for 20 images: {end_time - start_time:.4f} seconds")