简体中文 | [English](./README_en.md)

# 并行语音合成

[TOC]

## 新进展

- 2021/04/13 创建 [encoder](https://github.com/atomicoo/ParallelTTS/tree/encoder) 分支用于开发语音风格迁移模块！
- 2021/04/13 [softdtw](https://github.com/atomicoo/ParallelTTS/tree/softdtw) 分支 支持使用 [SoftDTW](https://arxiv.org/abs/1703.01541) 损失 训练模型！
- 2021/04/09 [wavegan](https://github.com/atomicoo/ParallelTTS/tree/wavegan) 分支 提供 [PWG](https://arxiv.org/abs/1910.11480) / [MelGAN](https://arxiv.org/abs/1910.06711) / [Multi-band MelGAN](https://arxiv.org/abs/2005.05106) 声码器！
- 2021/04/05 支持 [ParallelText2Mel](https://github.com/atomicoo/ParallelTTS/blob/main/models/parallel.py) + [MelGAN](https://arxiv.org/abs/1910.06711) 声码器！
- [ 关键信息 ]  [速度指标](#速度指标)，[合成样例](https://github.com/atomicoo/ParallelTTS/tree/main/samples/)，[网页演示](https://github.com/atomicoo/PTTS-WebAPP)，[一些问题](#一些问题)，[欢迎交流](#欢迎交流) ……

## 目录结构

```
.
|--- config/      # 配置文件
     |--- default.yaml
     |--- ...
|--- datasets/    # 数据处理
|--- encoder/     # 声纹编码器
     |--- voice_encoder.py
     |--- ...
|--- helpers/     # 一些辅助类
     |--- trainer.py
     |--- synthesizer.py
     |--- ...
|--- logdir/      # 训练过程保存目录
|--- losses/      # 一些损失函数
|--- models/      # 合成模型
     |--- layers.py
     |--- duration.py
     |--- parallel.py
|--- pretrained/  # 预训练模型（LJSpeech 数据集）
|--- samples/     # 合成样例
|--- utils/       # 一些通用方法
|--- vocoder/     # 声码器
     |--- melgan.py
     |--- ...
|--- wandb/       # Wandb 保存目录
|--- extract-duration.py
|--- extract-embedding.py
|--- LICENSE
|--- prepare-dataset.py  # 准备脚本
|--- README.md
|--- README_en.md
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
$ python synthesize.py \
  --checkpoint ./pretrained/ljspeech-parallel-epoch0100.pth \
  --melgan_checkpoint ./pretrained/ljspeech-melgan-epoch3200.pth \
  --input_texts ./samples/english/synthesize.txt \
  --outputs_dir ./outputs/
```

如果要合成其他语种的语音，需要通过 `--config` 指定相应的配置文件。

## 如何训练

**步骤（1）**：准备数据

```shell
$ python prepare-dataset.py
```

通过 `--config` 可以指定配置文件，默认的 [`default.yaml`](https://github.com/atomicoo/ParallelTTS/blob/main/config/default.yaml) 针对 [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) 数据集。

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

## 数据集

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)：英语，女性，22050 Hz，约 24 小时
- [LibriSpeech](https://www.openslr.org/12/)：英语，多说话人（仅使用 [train-clean-100](https://www.openslr.org/resources/12/train-clean-100.tar.gz) 部分），16000 Hz，总计约 1000 小时
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)：日语，女性，48000 Hz，约 10 小时
- [BiaoBei](https://www.data-baker.com/open_source.html)：普通话，女性，48000 Hz，约 12 小时
- [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)：韩语，女性，44100 Hz，约 12 小时
- [RuLS](https://www.openslr.org/96/)：俄语，多说话人（仅使用单一说话人音频），16000 Hz，总计约 98 小时
- [TWLSpeech](#)（非公开，质量较差）：藏语，女性（多说话人，音色相近），16000 Hz，约 23 小时

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

## 一些问题

- 在 [wavegan](https://github.com/atomicoo/ParallelTTS/tree/wavegan) 分支中，`vocoder` 代码取自 [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)，由于声学特征提取方式不兼容，需要进行转化，具体转化代码见[这里](https://github.com/atomicoo/ParallelTTS/blob/4eb44679271494f1d478da281ae474a07dfe77c6/synthesize.wave.py#L79-L85)。
- 普通话模型的文本输入选择拼音序列，因为 [BiaoBei](https://www.data-baker.com/open_source.html) 的原始拼音序列不包含标点、以及对齐模型训练不完全，所以合成语音的节奏会有点问题。
- 韩语模型没有专门训练对应的声码器，而是直接使用 LJSpeech（同为 22050 Hz）的声码器，可能稍微影响合成语音的质量。

## 参考资料

- [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron)
- [r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)
- [tugstugi/pytorch-dc-tts](https://github.com/tugstugi/pytorch-dc-tts)
- [janvainer/speedyspeech](https://github.com/janvainer/speedyspeech)
- [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
- [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda)
- [seungwonpark/melgan](https://github.com/seungwonpark/melgan)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## TODO

- [ ] 合成语音质量评估（MOS）
- [ ] 更多不同语种的测试
- [ ] 语音风格迁移（音色）

## 欢迎交流

- 微信号：Joee1995

- 企鹅号：793071559