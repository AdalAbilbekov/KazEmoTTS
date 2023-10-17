# KazEmoTTS

## Installation

First you need to build `monotonic_align` code:

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**:Python version is 3.9.13

## Pre-processing data for the training.

You need to download [KazEmo](https://drive.google.com/file/d/1jHzzqS58Te8xR1VqBl4dcpOCitsESi62/view?usp=share_link) corpus and customize it as in `filelists/all_spk` by running `data_preparation.py`:
```shell
python data_preparation.py -d provide the directory of KazEmo corpus
```

## Training stage.
To start the training, you need to provide path to the model configurations `configs/train_grad.json` and directory for checkpoints `logs/train_logs` specify your GPU.

```shell
CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
python train_EMA.py -c <configs/train_grad.json> -m <checkpoint>
```

## Inference.

### Pre-trained stage.
If you want to use pre-trained model you need to download [checkpoints](https://drive.google.com/file/d/1yfIOoVZEiFflh9494Ul6bUmktYvdM7XM/view?usp=share_link) for TTS model and vocoder.

To run inference use:
Create text file with sentences you want to synthesize like `filelists/inference_generated.txt`.
Specify `txt` file as follows: `text|emotion id|speaker id`.
Change path to the HiFi-Gan checkpoint in `inference_EMA.py`.
Apply classifier guidance level to 100 `-g`.
```shell
python inference_EMA.py -c <config> -m <checkpoint> -t <number-of-timesteps> -g <guidance-level> -f <path-for-text> -r <path-to-save-audios>
```


## References

* HiFi-GAN vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* GradTTS text2speech model, official github repository: [link](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
