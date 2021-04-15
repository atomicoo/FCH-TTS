[简体中文](./README.md) | English

# Parallel TTS

[TOC]

## What's New !

- 2021/04/13 create [encoder](https://github.com/atomicoo/ParallelTTS/tree/encoder) branch to dev the speech style transfer module!
- 2021/04/13 [softdtw](https://github.com/atomicoo/ParallelTTS/tree/softdtw) branch support [SoftDTW](https://arxiv.org/abs/1703.01541) loss!
- 2021/04/09 [wavegan](https://github.com/atomicoo/ParallelTTS/tree/wavegan) branch support [PWG](https://arxiv.org/abs/1910.11480) / [MelGAN](https://arxiv.org/abs/1910.06711) / [Multi-band MelGAN](https://arxiv.org/abs/2005.05106) vocoder!
- 2021/04/05 Support [ParallelText2Mel](https://github.com/atomicoo/ParallelTTS/blob/main/models/parallel.py) + [MelGAN](https://arxiv.org/abs/1910.06711) vocoder!
- [ Key Info ]  [Speed Indicator](#Speed), [Samples](https://github.com/atomicoo/ParallelTTS/tree/main/samples/), [Web Demo](https://github.com/atomicoo/PTTS-WebAPP), [Few Issues](#Few Issues), [Communication](#Communication) ......

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

Through `--config` to set config file, default ([`default.yaml`](https://github.com/atomicoo/ParallelTTS/blob/main/config/default.yaml)) is for [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

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

## Datasets

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): English, Female, 22050 Hz, ~24 h
- [LibriSpeech](https://www.openslr.org/12/): English, Multi-speakers (only use audios of [train-clean-100](https://www.openslr.org/resources/12/train-clean-100.tar.gz)),16000 Hz，total ~1000 h
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): Japanese, Female, 48000 Hz, ~10 h
- [BiaoBei](https://www.data-baker.com/open_source.html): Mandarin, Female, 48000 Hz, ~12 h
- [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset): Korean, Female, 44100 Hz, ~12 h
- [RuLS](https://www.openslr.org/96/): Russian, Multi-speakers (only use audios of single speaker), 16000 Hz, total ~98 h
- [TWLSpeech](#) (non-public, poor quality): Tibetan, Female (multi-speakers, sound similar), 16000 Hz，~23 h

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

Attention, no multiple tests, for reference only.

## Few Issues

- In [wavegan](https://github.com/atomicoo/ParallelTTS/tree/wavegan) branch, code of `vocoder` is from [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN). Since the method of acoustic feature extraction is not compatible, it needs to be transformed. See [here](https://github.com/atomicoo/ParallelTTS/blob/4eb44679271494f1d478da281ae474a07dfe77c6/synthesize.wave.py#L79-L85).
- The input of mandarin model is pinyin. Because of the lack of punctuations in [BiaoBei](https://www.data-baker.com/open_source.html)'s raw pinyin sequence and the incomplete alignment model training, there's something wrong with the rhythm of synthesized samples.
- I haven't trained a Korean vocoder specially, and just use the vocoder of LJSpeech (22050 Hz), which might slightly affect the quality of synthesized audio.

## References

- [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron)
- [r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)
- [tugstugi/pytorch-dc-tts](https://github.com/tugstugi/pytorch-dc-tts)
- [janvainer/speedyspeech](https://github.com/janvainer/speedyspeech)
- [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
- [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda)
- [seungwonpark/melgan](https://github.com/seungwonpark/melgan)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## TODO

- [ ] Synthetic speech quality assessment (MOS)
- [ ] More tests in different languages
- [ ] Speech style transfer (tone)

## Communication

- VX: Joee1995

- QQ: 793071559