# app.py
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append('./models')
sys.path.append('./utils')

from models.cnn_model import PetBreedCNN
from utils.data_loader import PetDataLoader

app = Flask(__name__)

# 全局变量
model = None
class_names = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model():
    """加载训练好的模型"""
    global model, class_names

    try:
        # 加载类别名称
        data_loader = PetDataLoader(data_dir='./data', download=False)
        _, _, _, class_names = data_loader.prepare_datasets(batch_size=1, img_size=128)

        print(f"加载 {len(class_names)} 个类别")

        # 初始化模型
        model = PetBreedCNN(num_classes=len(class_names)).to(device)

        # 检查模型文件
        model_path = './best_pet_breed_model.pth'

        if os.path.exists(model_path):
            print(f"找到模型文件: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            val_acc = checkpoint.get('val_acc', '未知')
            print(f"模型加载成功，验证准确率: {val_acc}%")
        else:
            print(f"警告: 模型文件不存在: {model_path}")
            print("将使用随机初始化的模型（仅用于演示）")
            model.eval()

        return True

    except Exception as e:
        print(f"加载模型失败: {e}")
        return False


def predict_image(image):
    """预测单张图像"""
    try:
        # 预处理图像
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 使用模型预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)

        # 获取top-3预测结果
        top_probs, top_indices = torch.topk(probabilities, 3)

        # 准备结果
        results = []
        for i in range(3):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            results.append({
                'breed': class_names[idx],
                'probability': float(prob * 100),
                'class_id': int(idx)
            })

        return results

    except Exception as e:
        print(f"预测错误: {e}")
        return []


@app.route('/')
def home():
    """主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    try:
        # 读取图像
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 预测
        predictions = predict_image(image)

        if not predictions:
            return jsonify({'error': '预测失败'}), 500

        # 将图像转换为base64用于显示
        buffered = io.BytesIO()

        # 调整图像大小用于预览
        preview_image = image.copy()
        preview_image.thumbnail((300, 300))
        preview_image.save(buffered, format="JPEG", quality=90)

        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/class_info', methods=['GET'])
def get_class_info():
    """获取所有类别信息"""
    if class_names is None:
        return jsonify({'error': '模型未加载'}), 500

    return jsonify({
        'classes': class_names,
        'num_classes': len(class_names)
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("宠物品种识别系统 - Web应用")
    print("=" * 60)
    print(f"使用设备: {device}")

    # 加载模型
    if load_model():
        print("✅ 模型加载完成")
    else:
        print("❌ 模型加载失败，请先运行 train.py 训练模型")
        exit(1)

    print(f"支持 {len(class_names)} 个品种")
    print("示例品种:", class_names[:5] if class_names else "无")

    print("\n启动服务器...")
    print("请访问: http://localhost:5000")
    print("=" * 60)

    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)