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
|--- vocoder/     # 声码器
     |--- melgan.py
     |--- ...
|--- models/      # 合成模型
     |--- layers.py
     |--- duration.py
     |--- parallel.py
|--- pretrained/  # 预训练模型（LJSpeech 数据集）
|--- samples/     # 合成样例
|--- utils/       # 一些通用方法
|--- wandb/       # Wandb 保存目录
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

## 训练日志

如果使用 [TensorBoardX](https://github.com/lanpa/tensorboardX)，则运行如下命令：

```
$ tensorboard --logdir logdir/[DIR]/
```

强烈推荐使用 [Wandb](https://wandb.ai/)（Weights & Biases），只需在上述训练命令中增加 `--enable_wandb` 选项。

## 质量评估

TODO：待补充

## 速度指标

**训练速度**：对于 [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) 数据集，设置批次尺寸为 64，可以在单张 8GB 显存的 GTX 1080 显卡上进行训练，训练 ~8h（~300 epochs）后即可合成质量较高的语音。

**合成速度**：以下测试在 CPU @ Intel Core i7-8550U / GPU @ NVIDIA GeForce MX150 下进行，每段合成音频在 8 秒左右（约 20 词）

| 批次尺寸 | Spec<br>(GPU) | Audio<br>(GPU) | Spec<br>(CPU) | Audio<br>(CPU) |
| -------- | ------------- | -------------- | ------------- | -------------- |
| 1        | 0.042         | 0.218          | 0.100         | 2.004          |
| 2        | 0.046         | 0.453          | 0.209         | 3.922          |
| 4        | 0.053         | 0.863          | 0.407         | 7.897          |
| 8        | 0.062         | 2.386          | 0.878         | 14.599         |

注意，没有进行多次测试取平均值，结果仅供参考。

## TODO

- [ ] 合成语音质量评估（MOS）
- [ ] 更多不同语种的测试
- [ ] 语音风格迁移（音色）

