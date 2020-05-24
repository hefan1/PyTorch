## 基于视觉注意力机制的细粒度图像分类方法 代码仓库
环境包版本 requirements.txt：
```
matplotlib==3.1.3
numpy==1.18.1
pandas==1.0.3
Pillow==7.0.0
pycparser==2.19
pyparsing==2.4.6
python-dateutil==2.8.1
pytz==2019.3
scikit-learn==0.22.1
scipy==1.4.1
six==1.14.0
torch==1.4.0
torchvision==0.5.0
```

## 使用方法
1. 下载数据集放到./data/目录下

2. 训练使用 python GCN_STN_BLN.py

3. 测试已训练好的model，从[这里]()下载三个state_dict，然后运行testCode.py
```bash
(PyTorch) shipeiqu1998@torch:~/LT$ python testCode.py 
Test accuracy is 85.95
```
