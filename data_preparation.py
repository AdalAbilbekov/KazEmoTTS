import kaldiio
import os
import librosa
from tqdm import tqdm
import glob
import json 
from shutil import copyfile
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='path to the emotional dataset')
    args = parser.parse_args()
    dataset_path = args.data
    filelists_path = 'filelists/all_spk'
    feats_scp_file = filelists_path + 'feats.scp'
    feats_ark_file = filelists_path + 'feats.ark'


    spks = ['805570882', '1263201035', '399172782']
    train_files = []
    eval_files = []
    for spk in spks:
        train_files += glob.glob(dataset_path + spk + "/train/*.wav")
        eval_files += glob.glob(dataset_path + spk + "/eval/*.wav")

    os.makedirs(filelists_path, exist_ok=True)

    with open(filelists_path + 'train_utts.txt', 'w', encoding='utf-8') as f:
        for wav_path in train_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(wav_name + '\n')
    with open(filelists_path + 'eval_utts.txt', 'w', encoding='utf-8') as f:
        for wav_path in eval_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(wav_name + '\n')

    with open(feats_scp_file, 'w') as feats_scp, \
        kaldiio.WriteHelper(f'ark,scp:{feats_ark_file},{feats_scp_file}') as writer:
        for root, dirs, files in tqdm(os.walk(dataset_path)):
            for file in files:
                if file.endswith('.wav'):
                    # Get the file name and relative path to the root folder
                    wav_path = os.path.join(root, file)
                    rel_path = os.path.relpath(wav_path, dataset_path)
                    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
                    signal, rate = librosa.load(wav_path)
                    spec = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=1024, 
                                                hop_length=256, win_length=1024, n_mels=80)
                    # Write the features to feats.ark and feats.scp
                    writer[wav_name] = spec
    
    emotions = [os.path.basename(x).split("_")[1] for x in glob.glob(dataset_path + '/**/**/*')]
    emotions = sorted(set(emotions))

    utt2spk = {}
    utt2emo = {}
    wavs = glob.glob(dataset_path + '**/**/*.wav')
    for wav_path in tqdm(wavs):
        wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        emotion =  emotions.index(wav_name.split("_")[1])
        if wav_path.split('/')[-3] == '1263201035':
            spk = 0 ## labels should start with 0
        elif wav_path.split('/')[-3] == '805570882':
            spk = 1
        else:
            spk = 2
        utt2spk[wav_name] = str(spk)
        utt2emo[wav_name] = str(emotion)
    utt2spk = dict(sorted(utt2spk.items()))
    utt2emo = dict(sorted(utt2emo.items()))

    with open(filelists_path + 'utt2emo.json', 'w') as fp:
        json.dump(utt2emo, fp,  indent=4)
    with open(filelists_path + 'utt2spk.json', 'w') as fp:
        json.dump(utt2spk, fp,  indent=4) 
    
    txt_files = sorted(glob.glob(dataset_path + '/**/**/*.txt'))
    count = 0
    utt2text = {}
    flag = False
    with open(filelists_path + 'text', 'w', encoding='utf-8') as write:
        for txt_path in txt_files:
            basename = os.path.basename(txt_path).replace('.txt', '')
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip("\n")
                utt2text[basename] = text                  
    utt2text = dict(sorted(utt2text.items()))

    vocab = set()
    with open(filelists_path + '/text', 'w', encoding='utf-8') as f:
        for x, y in utt2text.items():
            for c in y: vocab.add(c)
            f.write(x + ' ' +  y + '\n')
