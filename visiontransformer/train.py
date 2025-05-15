import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from VisionTransformer import VisionTransformer  # 你自己写的 ViT 模型类

def main():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 2. 数据预处理（CIFAR10: 32x32，ViT 需要224x224）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, num_workers=0)

    # 3. 模型定义
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        num_heads=3,
        num_layers=4,
        ffn_hidden=768,
        drop_out=0.1
    ).to(device)

    # 4. 损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 5. 训练与记录
    losses = []
    accuracies = []

    for epoch in range(2):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"Epoch {epoch+1} finished. Test Accuracy: {acc:.2f}%\n")

    # 6. 可视化保存
    plt.plot(losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("vit_train_loss.png")

    plt.figure()
    plt.plot(accuracies, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()
    plt.savefig("vit_test_accuracy.png")

    # 保存模型
    torch.save(model.state_dict(), "vit_cifar10.pth")

if __name__ == "__main__":
    main()
