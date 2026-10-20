"""CIFAR-10 CNN training，依照標準的 PyTorch training workflow。

比較一個 basic CNN 與一個小型 residual CNN（ResNet）。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---- Hyperparameter 與路徑設定 ----
BATCH_SIZE = 128          # 每個 batch 的圖片數量
EPOCHS = 30               # 訓練的 epoch 總數
LEARNING_RATE = 0.001     # 初始 learning rate
SEED = 42                 # Random seed，確保結果可重現
DATA_DIR = Path('data')   # CIFAR-10 dataset 下載位置
MODEL_DIR = Path('model') # Model weights 儲存位置
IMAGE_DIR = Path('image') # Training curve 圖片儲存位置


def get_dataloaders():
    """建立 training、validation、test 三個 DataLoader。"""
    # CIFAR-10 三個 channel（RGB）的 mean平均 與 std標準差，用來做 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    # Training 用的 transform：加入 data augmentation（random crop、horizontal flip）以減少 overfitting
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周 padding 4 個 pixel，再隨機裁回 32x32
        transforms.RandomHorizontalFlip(),     # 50% 機率左右翻轉
        transforms.ToTensor(),                 # 轉成 Tensor，pixel 值縮放到 [0, 1]
        transforms.Normalize(mean, std),       # Normalization
    ])
    # Validation／test 用的 transform：不做 data augmentation，只轉 Tensor 與 normalization
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 同一份 training set 載入兩次：一份套用 training transform，一份套用 eval transform（給 validation set 用）
    train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    val_data = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=eval_transform)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_transform)

    # 從 training set 切出 validation set（45000 training / 5000 validation）；test set 保留到最後才評估
    indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices, val_indices = indices[:45000], indices[45000:]
    # Training set 要 shuffle；validation 與 test set 不需要
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


class BasicCNN(nn.Module):
    """Basic CNN：三層 convolution + 兩層 fully connected layer。"""

    def __init__(self):
        super().__init__()
        # Feature extraction：convolution -> batch normalization -> ReLU -> max pooling
        # Convolution 後接 BatchNorm 時，convolution 的 bias 是多餘的，所以設 bias=False
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),    # 3x32x32 -> 32x32x32
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # -> 32x16x16
            nn.Conv2d(32, 64, 3, padding=1, bias=False),   # -> 64x16x16
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # -> 64x8x8
            nn.Conv2d(64, 128, 3, padding=1, bias=False),  # -> 128x8x8
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        # Classifier：flatten 後經過兩層 fully connected layer，輸出 10 個 class
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 8, 512),
                                        nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        return self.classifier(self.features(x))  # 輸出原始 logits，不做 softmax（CrossEntropyLoss 內含）


class ResidualBlock(nn.Module):
    """Residual block：輸出 = ReLU(F(x) + shortcut(x))。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path：兩層 3x3 convolution；第一層的 stride 可用來縮小 spatial size
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut：shape 相同時直接相加；
        # channel 數或 spatial size 改變時，用 1x1 convolution 把輸入調整成相同 shape
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels)))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Main path 加上 shortcut 後再做 ReLU
        return self.relu(out + self.shortcut(x))


class ResNetCNN(nn.Module):
    """小型 residual CNN：一層 convolution + 兩個 residual block + global average pooling。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),  # 3x32x32 -> 32x32x32
            nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32, 64, stride=2),             # -> 64x16x16
            ResidualBlock(64, 128, stride=2),            # -> 128x8x8
            nn.AdaptiveAvgPool2d((1, 1)),                # Global average pooling -> 128x1x1
        )
        # 只需要一層 fully connected layer，parameter 數量比 BasicCNN 少很多
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def train_one_epoch(model, loader, criterion, optimizer, device):
    """訓練一個 epoch，回傳平均 training loss。"""
    model.train()  # Training mode：BatchNorm 使用當前 batch 的 statistics
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()              # 1. 清除上一步的 gradient
        outputs = model(images)            # 2. Forward pass
        loss = criterion(outputs, labels)  # 3. 計算 loss
        loss.backward()                    # 4. Backpropagation 計算 gradient
        optimizer.step()                   # 5. 更新 parameters
        # loss 是 batch 平均，乘上 batch size 以便最後算整體平均
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()  # 評估時不需要計算 gradient，省記憶體也更快
def evaluate(model, loader, criterion, device):
    """在給定 dataset 上評估，回傳 (平均 loss, accuracy%)。"""
    model.eval()  # Evaluation mode：BatchNorm 改用 training 時累積的 running statistics
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        # 取 logits 最大值的 index 作為預測 class，計算答對數量
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)


def plot_metrics(history, name):
    """畫出 loss 與 accuracy curve 並存成圖片。"""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左圖：training loss vs validation loss（可觀察是否 overfitting）
    axes[0].plot(epochs, history['train_loss'], label='Train loss')
    axes[0].plot(epochs, history['val_loss'], label='Validation loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=f'{name} — Loss')
    axes[0].legend()
    # 右圖：validation accuracy
    axes[1].plot(epochs, history['val_acc'], label='Validation accuracy')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'{name} — Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f'{name.lower().replace(" ", "_")}_metrics.png')
    plt.close(fig)


def train_and_evaluate(model, name, train_loader, val_loader, test_loader, device):
    """完整訓練一個 model，保存最佳 weights，最後在 test set 上評估，回傳 test accuracy。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Multi-class classification 的 cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 每 10 個 epoch 把 learning rate 乘以 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = -1.0
    save_path = MODEL_DIR / f'{name.lower().replace(" ", "_")}_best.pth'

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'{name} | Epoch {epoch + 1:02d}/{EPOCHS} | '
              f'LR {optimizer.param_groups[0]["lr"]:.6f} | '
              f'Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val acc {val_acc:.2f}%')
        # Validation accuracy 創新高時，保存這個 checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
        scheduler.step()  # 每個 epoch 結束後更新 learning rate

    # 載入 validation 表現最好的 checkpoint，只在 test set 上評估一次
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'{name} | Best val acc {best_acc:.2f}% | Test loss {test_loss:.4f} | Test acc {test_acc:.2f}%')
    plot_metrics(history, name)
    return test_acc


def main():
    torch.manual_seed(SEED)
    # 選擇 device：優先 NVIDIA GPU (cuda)，其次 Apple Silicon GPU (mps)，否則用 CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    train_loader, val_loader, test_loader = get_dataloaders()
    results = {}
    # 依序訓練兩個 model 並比較 test accuracy
    for name, model in [('Basic CNN', BasicCNN()), ('ResNet CNN', ResNetCNN())]:
        results[name] = train_and_evaluate(model, name, train_loader, val_loader, test_loader, device)
    print('\nFinal test accuracy:', results)


if __name__ == '__main__':
    main()
