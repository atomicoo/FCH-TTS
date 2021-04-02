# 并行语音合成

[TOC]

## 目录结构

```
.
|--- config/      # 配置文件
     |--- default.yaml
     |--- ...
|--- datasets/    # 数据处理
|--- helpers/     # 一些辅助类
     |--- trainer.py
     |--- synthesizer.py
     |--- ...
|--- logdir/      # 训练过程保存目录
|--- losses/      # 一些损失函数
|--- melgan/      # 声码器
     |--- generator.py
     |--- ...
|--- models/      # 合成模型
     |--- layers.py
     |--- duration.py
     |--- parallel.py
|--- pretrained/  # 预训练模型（LJSpeech 数据集）
|--- samples/     # 合成样例
|--- utils/       # 一些通用方法
|--- LICENSE
|--- prepare-dataset.py  # 准备脚本
|--- extract-duration.py
|--- README.md
|--- requirements.txt    # 依赖文件
|--- synthesize.py       # 合成脚本
|--- train-duration.py   # 训练脚本
|--- train-parallel.py
```

## 合成样例

部分合成样例见[这里](https://github.com/atomicoo/ParallelTTS/tree/main/samples/)。

## 预训练

部分预训练模型见[这里](https://github.com/atomicoo/ParallelTTS/tree/main/pretrained/)。

## 快速开始

**步骤（1）**：克隆仓库

```shell
$ git clone https://github.com/atomicoo/ParallelTTS.git
```

**步骤（2）**：安装依赖

```shell
$ conda create -n ParallelTTS python=3.7.9
$ conda activate ParallelTTS
$ pip install -r requirements.txt
```

**步骤（3）**：合成语音

```shell
$ python synthesize.py
```

## 如何训练

**步骤（1）**：准备数据

```shell
$ python prepare-dataset.py
```

通过 `--config` 可以指定配置文件，默认的 `default.yaml` 针对 LJSpeech 数据集。

**步骤（2）**：训练对齐模型

```shell
$ python train-duration.py 
```

**步骤（3）**：提取持续时间

```shell
$ python extract-duration.py
```

通过 `--ground_truth` 可以指定是否利用对齐模型生成 Ground-Truth 声谱图。

**步骤（4）**：训练合成模型

```shell
$ python train-parallel.py
```

通过 `--ground_truth` 可以指定是否使用 Ground-Truth 声谱图进行模型训练。

