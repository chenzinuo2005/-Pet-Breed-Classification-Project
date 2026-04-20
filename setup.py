#!/usr/bin/env python3
"""
自动下载依赖并配置项目
"""

import subprocess
import sys
import os

def run_command(cmd, desc):
    print(f"\n{'='*50}")
    print(f"{desc}...")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {desc} 失败")
        sys.exit(1)
    print(f"[OK] {desc} 完成")

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 安装依赖
    run_command(
        f'pip install -r "{project_dir}/requirements.txt"',
        "安装 Python 依赖"
    )

    # 2. 下载数据集
    run_command(
        f'python "{project_dir}/download_data.py"',
        "下载 Oxford-IIIT Pet Dataset"
    )

    print("\n" + "="*50)
    print("项目初始化完成！")
    print("="*50)
    print("\n下一步：")
    print("  训练模型: python train.py")
    print("  启动服务: python app.py")

if __name__ == "__main__":
    main()
