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