# -Pet-Breed-Classification-Project
基于 PyTorch 与 Flask 的宠物品种图像分类系统，支持识别 37 种猫狗品种。
  功能特性

  - Web 推理接口：用户上传宠物图片，快速返回 Top-3 品种预测及置信度
  - 自定义 CNN 模型：4 层卷积神经网络，含批量归一化与 Dropout 正则化
  - 完整训练流程：支持学习率预热、余弦退火调度、梯度裁剪与早停
  - 数据增强：随机裁剪、翻转、旋转、颜色抖动、随机擦除等策略提升泛化能力
  - 可视化分析：训练曲线、混淆矩阵、误分类样本分析

  技术栈

  
     类别              工具            
  
   深度学习   PyTorch, torchvision        
  
   Web 框架   Flask                       
  
   数据处理   Pillow, numpy, scikit-learn 
  
   可视化     matplotlib, seaborn         
  

  项目结构

  app.py              # Flask Web 应用（主入口）
  train.py            # 模型训练脚本
  evaluate.py         # 模型评估脚本
  download_data.py    # 数据集下载工具
  models/cnn_model.py # CNN 模型定义
  utils/data_loader.py  # 数据加载与增强
  utils/visualize.py    # 训练可视化
  templates/index.html  # 前端页面

  这是一个端到端的机器学习项目，涵盖数据获取、模型训练、Web 部署与结果分析，适合作为图像分类任务的学习范例。
