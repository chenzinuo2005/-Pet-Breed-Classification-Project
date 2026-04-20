# train.py - 完整修复版本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# 添加项目路径
sys.path.append('./models')
sys.path.append('./utils')

from models.cnn_model import PetBreedCNN
from utils.data_loader import PetDataLoader
from utils.visualize import plot_training_history


class PetBreedTrainer:
    def __init__(self, num_classes=37, img_size=128, device=None):
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.num_classes = num_classes

        # 创建模型，传入图像尺寸
        self.model = PetBreedCNN(num_classes=num_classes, img_size=img_size).to(self.device)

        # 使用Adam优化器，设置合适的学习率
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减

        # 使用余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        # 学习率预热
        self.warmup_epochs = 5
        self.base_lr = 0.001

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def adjust_learning_rate(self, epoch):
        """学习率调整"""
        if epoch < self.warmup_epochs:
            # 学习率预热
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 使用调度器
            self.scheduler.step()

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """验证方法"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=100, save_path='./best_pet_breed_model.pth'):
        """完整训练过程"""
        best_val_acc = 0.0
        patience = 10  # 早停耐心值
        patience_counter = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 调整学习率
            self.adjust_learning_rate(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, save_path)
                print(f'✅ Model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1

            # 打印epoch结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')

            # 早停机制
            if patience_counter >= patience:
                print(f'\n⚠️  早停触发，连续{patience}轮验证准确率未提升')
                break

        print(f'\nBest Validation Accuracy: {best_val_acc:.2f}%')
        return self.history

    def test(self, test_loader):
        """测试模型"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        test_acc = 100. * correct / total
        print(f'\nTest Accuracy: {test_acc:.2f}%')

        return test_acc, all_preds, all_labels


def main():
    # 参数设置
    BATCH_SIZE = 16
    IMG_SIZE = 128
    EPOCHS = 100



    # 数据加载
    print("\n1. 加载数据...")
    data_loader = PetDataLoader(data_dir='./data', download=False)

    try:
        # 使用稳定的图像尺寸
        train_loader, val_loader, test_loader, class_names = data_loader.prepare_datasets(
            batch_size=BATCH_SIZE, img_size=IMG_SIZE
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("尝试重新下载数据集...")

        # 尝试手动下载
        import shutil
        if os.path.exists('./data/raw'):
            shutil.rmtree('./data/raw')

        data_loader = PetDataLoader(data_dir='./data', download=True)
        train_loader, val_loader, test_loader, class_names = data_loader.prepare_datasets(
            batch_size=BATCH_SIZE, img_size=IMG_SIZE
        )

    print(f"\n2. 数据集信息:")
    print(f"   类别数量: {len(class_names)}")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    print(f"   示例类别: {class_names[:5]}")

    # 创建训练器，传入图像尺寸
    trainer = PetBreedTrainer(num_classes=len(class_names), img_size=IMG_SIZE)

    # 训练模型
    print("\n3. 开始训练模型...")
    print("   注意：这将需要一些时间，请耐心等待...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=EPOCHS,
        save_path='./best_pet_breed_model.pth'
    )

    # 绘制训练历史
    plot_training_history(history)

    # 测试模型
    print("\n4. 测试模型...")
    test_acc, predictions, true_labels = trainer.test(test_loader)

    # 显示结果
    print("=" * 60)
    if test_acc < 85:
        print(f"⚠️  测试准确率: {test_acc:.2f}% (低于85%)")
        print("\n建议进一步优化:")
        print("  1. 增加训练轮数")
        print("  2. 使用学习率调整")
        print("  3. 增加数据增强")
    else:
        print(f"✅ 测试准确率: {test_acc:.2f}% (达到目标!)")
        print(f"\n🎉 恭喜！模型训练成功！")
    print("=" * 60)

    return trainer, history, test_acc


if __name__ == '__main__':
    trainer, history, test_acc = main()