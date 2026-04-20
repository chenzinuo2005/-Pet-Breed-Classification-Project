# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os
import sys

# 添加项目路径
sys.path.append('./models')
sys.path.append('./utils')

from models.cnn_model import PetBreedCNN
from utils.data_loader import PetDataLoader
from utils.visualize import plot_confusion_matrix, plot_sample_predictions


def evaluate_model(model, test_loader, class_names, device='cuda'):
    """评估模型并生成详细报告"""
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(inputs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )

    return {
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels),
        'confusion_matrix': cm,
        'classification_report': report,
        'images': np.array(all_images),
        'probabilities': np.array(all_probs)
    }


def analyze_misclassifications(results, test_loader, class_names, top_n=10):
    """分析错误分类案例"""
    predictions = results['predictions']
    true_labels = results['true_labels']
    probabilities = results['probabilities']

    # 找到错误分类的样本
    misclassified_idx = np.where(predictions != true_labels)[0]

    if len(misclassified_idx) == 0:
        print("No misclassifications found!")
        return

    print(f"\nFound {len(misclassified_idx)} misclassifications")
    print("\nTop misclassified cases:")

    # 获取测试集数据
    test_data = test_loader.dataset

    # 显示前几个错误分类
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i, idx in enumerate(misclassified_idx[:10]):
        # 获取图像和标签
        image, true_label = test_data[idx]
        pred_label = predictions[idx]

        # 反归一化图像
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # 显示图像
        axes[i].imshow(image)
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('misclassifications.png')
    plt.show()

    # 分析哪些类别最容易混淆
    misclassification_pairs = []
    for idx in misclassified_idx:
        true_cls = class_names[true_labels[idx]]
        pred_cls = class_names[predictions[idx]]
        misclassification_pairs.append((true_cls, pred_cls))

    # 创建DataFrame进行分析
    df_misclass = pd.DataFrame(misclassification_pairs, columns=['True', 'Predicted'])
    confusion_counts = df_misclass.groupby(['True', 'Predicted']).size().reset_index(name='Count')

    print("\nMost common misclassifications:")
    print(confusion_counts.sort_values('Count', ascending=False).head(10))


def main():
    # 参数
    BATCH_SIZE = 32
    IMG_SIZE = 128

    # 加载数据
    data_loader = PetDataLoader(data_dir='./data', download=False)
    _, _, test_loader, class_names = data_loader.prepare_datasets(
        batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )

    # 加载训练好的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PetBreedCNN(num_classes=len(class_names)).to(device)

    # 检查模型文件是否存在
    model_path = './best_pet_breed_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        print("请先运行 train.py 训练模型")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("=" * 60)
    print("宠物品种识别系统 - 评估阶段")
    print("=" * 60)
    print(f"模型加载完成，来自第 {checkpoint['epoch']} 轮")
    print(f"验证准确率: {checkpoint['val_acc']:.2f}%")

    # 评估模型
    print("\n评估模型...")
    results = evaluate_model(model, test_loader, class_names, device)

    # 计算测试准确率
    accuracy = np.mean(results['predictions'] == results['true_labels']) * 100
    print(f"\n测试准确率: {accuracy:.2f}%")

    # 显示混淆矩阵
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        normalize=True
    )

    # 分析错误分类
    analyze_misclassifications(results, test_loader, class_names)

    # 显示样本预测
    plot_sample_predictions(model, test_loader, class_names, device)

    # 显示分类报告
    report_df = pd.DataFrame(results['classification_report']).transpose()
    print("\n分类报告:")
    print(report_df[['precision', 'recall', 'f1-score', 'support']].round(2))


if __name__ == '__main__':
    main()