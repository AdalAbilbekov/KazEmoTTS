# KazEmoTTS
### Organizing Data
Files that specify dataset (i.e. comprised of utterances) should be put in `filelists/${name}`. `{name}` is any subdirectory.

Explanation for the files in `filelists/example`:
* `train_utts.txt` and `val_utts.txt`: files containing training and validation utterances.
* `feats.scp`: We use Kaldi-style IO to save and load features (in this case, mel-spectrograms). For example, `feats.scp` specifies the acoustic features for each utterance.
You can turn to https://github.com/nttcslab-sp/kaldiio for more information.
* `text`: each line consists of utterance ID and corresponding phoneme sequence.
* `utt2spk.json`, `utt2emo.json`: json files for utterance-speaker and utterance-emotion pair.
* `utt2dummy_emo.json`: all utterances share the same emotion. This is for unconditional acoustic model.
* `phn_duration`: each lines consists of utterance ID and corresponding phoneme duration. This is acquired by Kaldi, but you can seek for other methods. Also, you can alternatively let the model learn duration by itself.
* `phones.txt`: all phones and their indexes.

Then we specify these files in the `configs/${name}.json`. Some entries may also need to be changed, like number of speakers.

### Training of unconditional acoustic model (GradTTS)
See `train.py`. Example usage is 
```shell
python train.py -c configs/${name}.json -m ${name}
```
It will save models in `logs/${name}`.

Also see `train_EMA.py` if you want to enable EMA training.

### Training of classifier
See `train_classifier.py`. Usage is the same I think. 
It loads `grad_uncond.pt` from the `logs/${name}` directory. 

### Inference
See `inference.py` and `inference_cls_guidance_two_mixture.py`. The arguments are specified in `utils.get_hparams_decode` and `utils.get_hparams_decode_two_mixture`.
You need to provide `-c` and `-m` flags the same as training. There are various other arguments in these functions.

Finally, the generated features (mel spectrograms) will be saved in `synthetic_wavs/${name}/xxx` where `xxx` depends on inference type. 
The features are also saved in kaldi-style `feats.ark` and `feats.scp`.  
___________________________________________________________________________________________________________________________________________________________________________________
# Grad-TTS

Official implementation of the Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

**Authors**: Vadim Popov\*, Ivan Vovk\*, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov.

<sup>\*Equal contribution.</sup>

## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.

## Inference

You can download Grad-TTS and HiFi-GAN checkpoints trained on LJSpeech* and Libri-TTS datasets (22kHz) from [here](https://drive.google.com/drive/folders/1grsfccJbmEuSBGQExQKr3cVxNV0xEOZ7?usp=sharing).

***Note**: we open-source 2 checkpoints of Grad-TTS trained on LJSpeech. They are the same models but trained with different positional encoding scale: **x1** (`"grad-tts-old.pt"`, ICML 2021 sumbission model) and **x1000** (`"grad-tts.pt"`). To use the former set `params.pe_scale=1` and to use the latter set `params.pe_scale=1000`. Libri-TTS checkpoint was trained with scale **x1000**.

Put necessary Grad-TTS and HiFi-GAN checkpoints into `checkpts` folder in root Grad-TTS directory (note: in `inference.py` you can change default HiFi-GAN path).

1. Create text file with sentences you want to synthesize like `resources/filelists/synthesis.txt`.
2. For single speaker set `params.n_spks=1` and for multispeaker (Libri-TTS) inference set `params.n_spks=247`.
3. Run script `inference.py` by providing path to the text file, path to the Grad-TTS checkpoint, number of iterations to be used for reverse diffusion (default: 10) and speaker id if you want to perform multispeaker inference:
    ```bash
    python inference.py -f <your-text-file> -c <grad-tts-checkpoint> -t <number-of-timesteps> -s <speaker-id-if-multispeaker>
    ```
4. Check out folder called `out` for generated audios.

You can also perform *interactive inference* by running Jupyter Notebook `inference.ipynb` or by using our [Google Colab Demo](https://colab.research.google.com/drive/1YNrXtkJQKcYDmIYJeyX8s5eXxB4zgpZI?usp=sharing).

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. For single speaker training refer to `jspeech` filelists and to `libri-tts` filelists for multispeaker.
2. Set experiment configuration in `params.py` file.
3. Specify your GPU device and run training script:
    ```bash
    export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
    python train.py  # if single speaker
    python train_multi_speaker.py  # if multispeaker
    ```
4. To track your training process run tensorboard server on any available port:
    ```bash
    tensorboard --logdir=YOUR_LOG_DIR --port=8888
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).
