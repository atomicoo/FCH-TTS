[简体中文](./README.md) | English

# Parallel TTS

[TOC]

## Repo Structure

```
.
|--- config/      # config file
     |--- default.yaml
     |--- ...
|--- datasets/    # data process
|--- encoder/     # voice encoder
     |--- voice_encoder.py
     |--- ...
|--- helpers/     # some helpers
     |--- trainer.py
     |--- synthesizer.py
     |--- ...
|--- logdir/      # training log directory
|--- losses/      # loss function
|--- models/      # synthesizor
     |--- layers.py
     |--- duration.py
     |--- parallel.py
|--- pretrained/  # pretrained (LJSpeech dataset)
|--- samples/     # synthesized samples
|--- utils/       # some common utils
|--- vocoder/     # vocoder
     |--- melgan.py
     |--- ...
|--- wandb/       # Wandb save directory
|--- extract-duration.py
|--- extract-embedding.py
|--- LICENSE
|--- prepare-dataset.py  # prepare dataset
|--- README.md
|--- requirements.txt    # dependencies
|--- synthesize.py       # synthesize script
|--- train-duration.py   # train script
|--- train-parallel.py
```

## Samples

[Here](https://github.com/atomicoo/ParallelTTS/tree/main/samples/) are some synthesized samples.

## Pretrained

[Here](https://github.com/atomicoo/ParallelTTS/tree/main/pretrained/) are some pretrained models.

## Quick Start

**Step (1)**：clone repo

```shell
$ git clone https://github.com/atomicoo/ParallelTTS.git
```

**Step (2)**：install dependencies

```shell
$ conda create -n ParallelTTS python=3.7.9
$ conda activate ParallelTTS
$ pip install -r requirements.txt
```

**Step (3)**：synthesize audio

```shell
$ python synthesize.py
```

## Training

**Step (1)**：prepare dataset

```shell
$ python prepare-dataset.py
```

Through `--config` to set config file, default (`default.yaml`) is for LJSpeech dataset.

**Step (2)**：train alignment model

```shell
$ python train-duration.py 
```

**Step (3)**：extract durations

```shell
$ python extract-duration.py
```

Through `--ground_truth` to set weather generating ground-truth spectrograms or not。

**Step (4)**：train synthesize model

```shell
$ python train-parallel.py
```

Through `--ground_truth` to set weather training model by ground-truth spectrograms。

## Training Log

if use [TensorBoardX](https://github.com/lanpa/tensorboardX), run this: 

```
$ tensorboard --logdir logdir/[DIR]/
```

It is highly recommended to use [Wandb](https://wandb.ai/)（Weights & Biases）, just set `--enable_wandb` when training。

## Quality

TODO: to be added.

## Speed

**Speed of Training**：[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset, batch size = 64, training on 8GB GTX 1080 GPU, elapsed ~8h (~300 epochs).

**Speed of Synthesizing**：test under CPU @ Intel Core i7-8550U / GPU @ NVIDIA GeForce MX150, 8s per synthesized audio (about 20 words)

| Batch Size | Spec<br>(GPU) | Audio<br>(GPU) | Spec<br>(CPU) | Audio<br>(CPU) |
| ---------- | ------------- | -------------- | ------------- | -------------- |
| 1          | 0.042         | 0.218          | 0.100         | 2.004          |
| 2          | 0.046         | 0.453          | 0.209         | 3.922          |
| 4          | 0.053         | 0.863          | 0.407         | 7.897          |
| 8          | 0.062         | 2.386          | 0.878         | 14.599         |

Attention, 

## TODO

- [ ] Synthetic speech quality assessment (MOS)
- [ ] More tests in different languages
- [ ] Speech style transfer (tone)

## Communication

- VX: Joee1995

- QQ: 793071559