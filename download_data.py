# download_data.py
import os
import tarfile
import requests
from tqdm import tqdm


def download_dataset():
    """下载Oxford-IIIT Pet Dataset"""

    # 创建目录
    os.makedirs('./data/raw', exist_ok=True)

    # 数据集URL
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    urls = {
        'images': base_url + 'images.tar.gz',
        'annotations': base_url + 'annotations.tar.gz'
    }

    def download_file(url, filename):
        """下载单个文件"""
        print(f"下载: {filename}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))

    # 下载并解压
    for name, url in urls.items():
        filename = f'./data/raw/{name}.tar.gz'

        if not os.path.exists(filename.replace('.tar.gz', '')):
            # 下载
            download_file(url, filename)

            # 解压
            print(f"解压: {filename}")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall('./data/raw')

            # 删除压缩包
            os.remove(filename)
            print(f"完成: {name}")
        else:
            print(f"已存在: {name}")

    print("\n数据集下载完成！")


if __name__ == '__main__':
    download_dataset()