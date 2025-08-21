import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
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
    
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validate'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def inference(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    shown = 0
    max_show = 30
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def denormalize(img, mean, std):
        img = img.cpu().clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img
    # 30개 이미지, 예측값, 실제값을 하나의 창에 출력
    imgs_to_show = []
    preds_to_show = []
    labels_to_show = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Inference'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 실제 레이블이 0 또는 1인 것만 저장, 최대 30개
            for i in range(images.size(0)):
                if shown < max_show and labels[i].item() in [0, 1]:
                    imgs_to_show.append(denormalize(images[i], mean, std))
                    preds_to_show.append(predicted[i].item())
                    labels_to_show.append(labels[i].item())
                    shown += 1
                if shown >= max_show:
                    break
            if shown >= max_show:
                break
    if imgs_to_show:
        plt.figure(figsize=(20, 12))
        for i in range(len(imgs_to_show)):
            plt.subplot(6, 5, i+1)
            plt.imshow(imgs_to_show[i].permute(1, 2, 0))
            plt.title(f"Pred: {preds_to_show[i]} / Label: {labels_to_show[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    accuracy = 100 * correct / total
    return accuracy

def main(inference_only=False):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # -15 ~ +15도 사이의 무작위 회전
        transforms.RandomRotation(degrees=15),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = './data'
    test_dir = 'data/mnist_test'
    batch_size = 64

    train_dataset = datasets.MNIST(root=data_dir,
                                train=True,
                                download=True,
                                transform=train_transform)
    val_dataset = datasets.MNIST(root=data_dir,
                                train=False,
                                download=True,
                                transform=validation_transform)
    test_dataset = CustomDataset(test_dir,
                                transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # MNIST 분류를 위해 FC수정
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epoch = 50
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    model_save_dir = 'data/mnist_models_2'
    plot_save_dir = 'data/mnist_graphs_2'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    best_model_path = os.path.join(model_save_dir, 'best_model.pth')

    if not inference_only:
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_accuracy = 0.0
        plt.ion()

        print(f"Start Training... Using device: {device}")
        for epoch in range(num_epoch):
            print(f'\nEpoch: {epoch+1}/{num_epoch}')
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}%")

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            model_save_path = os.path.join(model_save_dir, f'model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            if best_accuracy < val_acc:
                best_accuracy = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} with acc: {best_accuracy:.4f}%")

            plt.figure(1, figsize=(15, 5))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.grid(True)
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.grid(True)
            plt.title('Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc.', marker='o')
            plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc.', marker='o')
            plt.legend()

            plt.tight_layout()
            plt.pause(0.01)
            plot_save_path = os.path.join(plot_save_dir, f'graph_epoch{epoch+1}.png')
            plt.savefig(plot_save_path)
            print(f"Graph saved to {plot_save_path}")

        plt.ioff()
        print(f"\nTraining finished.")

    print('\nStart Inferencing...')
    model.load_state_dict(torch.load(best_model_path))
    print(f"model loaded from {best_model_path}")
    final_accuracy = inference(model, test_loader, device)
    print(f"Final Accuracy: {final_accuracy:.4f}%")

if __name__ == '__main__':
    main(inference_only=True)