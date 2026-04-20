# utils/data_loader.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import tarfile
import requests
from tqdm import tqdm
import numpy as np


class OxfordPetDataset(Dataset):
    """自定义Oxford-IIIT Pet Dataset加载器"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 数据根目录
            transform: 数据变换
        """
        self.root_dir = os.path.join(root_dir, 'images')
        self.transform = transform

        # 读取所有图像文件
        self.image_files = [f for f in os.listdir(self.root_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not self.image_files:
            raise FileNotFoundError(f"在 {self.root_dir} 中没有找到图像文件")

        # 从文件名提取类别
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        for img_file in self.image_files:
            # 文件名格式: breed_id.jpg (例如: Abyssinian_1.jpg)
            breed = img_file.rsplit('_', 1)[0]

            if breed not in self.class_to_idx:
                self.class_to_idx[breed] = len(self.classes)
                self.classes.append(breed)

            class_id = self.class_to_idx[breed]
            img_path = os.path.join(self.root_dir, img_file)
            self.samples.append((img_path, class_id))

        print(f"找到 {len(self.samples)} 张图像, {len(self.classes)} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class PetDataLoader:
    def __init__(self, data_dir='./data', download=True):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
            download: 是否自动下载数据集
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')

        # 创建目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        if download:
            self.download_dataset()

    def download_dataset(self):
        """下载Oxford-IIIT Pet Dataset"""
        print("正在下载Oxford-IIIT Pet Dataset...")

        # 检查是否已存在数据集
        images_dir = os.path.join(self.raw_dir, 'images')
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            print("数据集已存在，跳过下载")
            return

        # 数据集URL
        base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
        urls = {
            'images': base_url + 'images.tar.gz',
            'annotations': base_url + 'annotations.tar.gz'
        }

        def download_file(url, filename):
            """下载单个文件"""
            print(f"下载: {os.path.basename(filename)}")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                with open(filename, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            if data:
                                f.write(data)
                                pbar.update(len(data))
                return True
            except Exception as e:
                print(f"下载失败: {e}")
                return False

        # 下载并解压
        for name, url in urls.items():
            filename = os.path.join(self.raw_dir, f"{name}.tar.gz")
            target_dir = filename.replace('.tar.gz', '')

            if not os.path.exists(target_dir):
                if download_file(url, filename):
                    # 解压
                    print(f"解压: {os.path.basename(filename)}")
                    try:
                        with tarfile.open(filename, 'r:gz') as tar:
                            tar.extractall(self.raw_dir)
                        print(f"完成: {name}")

                        # 删除压缩包
                        os.remove(filename)
                    except Exception as e:
                        print(f"解压失败: {e}")
                else:
                    print(f"跳过 {name} 的下载")
            else:
                print(f"已存在: {name}")

        print("数据集下载完成!")

    def prepare_datasets(self, batch_size=32, img_size=128):
        """
        准备训练、验证和测试数据集

        Args:
            batch_size: 批次大小
            img_size: 图像尺寸
        """

        # 增强的数据增强和预处理
        train_transform = transforms.Compose([
            # 随机调整图像大小并裁剪
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),

            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),

            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.1),

            # 随机旋转
            transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),

            # 颜色变换
            transforms.ColorJitter(
                brightness=0.3,  # 亮度调整
                contrast=0.3,  # 对比度调整
                saturation=0.3,  # 饱和度调整
                hue=0.1  # 色相调整
            ),

            # 随机仿射变换
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 随机平移
                scale=(0.9, 1.1),  # 随机缩放
                shear=10  # 随机剪切
            ),

            # 随机透视变换
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

            # 随机高斯模糊
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            # 转换为张量并归一化
            transforms.ToTensor(),

            # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

            # 随机擦除（Cutout）
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])

        # 验证和测试集的transform（只有基本预处理）
        val_test_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),  # 稍微放大一点
            transforms.CenterCrop(img_size),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 创建完整数据集（不应用transform）
        full_dataset = OxfordPetDataset(
            root_dir=self.raw_dir,
            transform=None  # 不在这里应用transform
        )

        # 按7:2:1划分数据集
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        print(f"数据集统计:")
        print(f"  总样本数: {total_size}")
        print(f"  训练集: {train_size}")
        print(f"  验证集: {val_size}")
        print(f"  测试集: {test_size}")

        # 划分数据集
        train_indices, val_indices, test_indices = random_split(
            list(range(total_size)), [train_size, val_size, test_size]
        )

        # 创建带有transform的数据集
        train_dataset = self._create_subset_with_transform(
            full_dataset, train_indices, train_transform
        )
        val_dataset = self._create_subset_with_transform(
            full_dataset, val_indices, val_test_transform
        )
        test_dataset = self._create_subset_with_transform(
            full_dataset, test_indices, val_test_transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )



        return train_loader, val_loader, test_loader, full_dataset.classes

    def _create_subset_with_transform(self, dataset, indices, transform):
        """创建带有transform的数据子集"""

        class SubsetWithTransform(Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                # 获取原始图像和标签
                img_idx = self.indices[idx]
                image, label = self.dataset[img_idx]  # 这里得到的是PIL Image

                # 应用transform
                if self.transform:
                    image = self.transform(image)

                return image, label

        return SubsetWithTransform(dataset, indices, transform)