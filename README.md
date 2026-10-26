# 基于 NumPy 手写实现的轻量级 CNN 手写数字分类系统

本项目是计算机视觉课程大作业。代码只使用 NumPy 完成卷积层、ReLU、最大池化、全连接层、Softmax 交叉熵、反向传播和 Adam 更新，不调用 PyTorch、TensorFlow 或现成分类模型。

## 项目结构

```text
src/
  cnn_numpy.py      CNN 前向与反向传播
  download_data.py  下载真实 MNIST 数据集
  train.py          训练、评估与图表输出
  app.py            本地浏览器演示系统
web/                可交互前端
artifacts/          训练后生成的权重、指标与图表
```

## 环境

- Python 3.10 或更高版本
- NumPy
- Pillow

```bash
python -m pip install -r requirements.txt
```

## 一键复现

```bash
python src/download_data.py
python src/train.py --epochs 10 --train-size 12000 --val-size 2000 --test-size 2000
python src/app.py --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。可以切换真实 MNIST 测试样本，也可以在画布中手写数字后执行识别。

![在线识别界面](docs/inference-ui.png)

## 本次实验结果

- 测试集准确率：96.40%
- 测试样本：2,000 张，其中 72 张误分类
- 模型参数量：2,970
- 批量推理速度：85,592 FPS（本机 CPU、2,000 张批量测试）
- 训练耗时：7.30 秒（12,000 张训练集、10 个 epoch）

![训练与评估结果](docs/results-ui.png)

## 实验设定

- 固定随机种子：42
- 输入：MNIST 28×28 灰度图，平均池化预处理为 14×14
- 结构：8 个 3×3 卷积核，ReLU，2×2 最大池化，288×10 全连接层
- 数据增强：随机平移 1 像素与轻微高斯噪声
- 优化器：手写 Adam
- 损失函数：Softmax 交叉熵

训练完成后，`artifacts/metrics.json` 记录全部超参数、曲线数据、测试准确率、逐类准确率、混淆矩阵和推理速度。

## 数据来源

MNIST 数据通过 Keras 官方公开镜像下载：
`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`

## 成员分工

- 张三（1111111）：CNN 算法设计、前向与反向传播实现、训练调优、代码仓库维护
- 李四（2222222）：数据处理与增强、测试评估、误判分析、报告与答辩材料
