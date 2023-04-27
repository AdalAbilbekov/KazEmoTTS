import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
# from model import GradTTSWithEmo
from torch.utils.data import DataLoader
from text import text_to_sequence, cmudict
from text.symbols import symbols
import utils
from kaldiio import WriteHelper
import os
from tqdm import tqdm


def evaluate(hps, args, ckpt, feats_dir):
    logger = utils.get_logger(hps.model_dir, "inference.log")
    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    torch.manual_seed(hps.train.seed)  # NOTE: control seed
    train_dataset, collate_fn, model = utils.get_correct_class(hps)
    val_dataset, _, _ = utils.get_correct_class(hps, train=False)
    batch_collate = collate_fn
    train_loader = DataLoader(dataset=train_dataset, batch_size=1,
                              collate_fn=batch_collate, drop_last=True,
                              num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4
    val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                            collate_fn=batch_collate, drop_last=True,
                            num_workers=4, shuffle=False)
    model = model(**hps.model).to(device)
    utils.load_checkpoint(ckpt, model, None)
    print(f"Loaded checkpoint from {ckpt}")
    _ = model.cuda().eval()
    print(f'Number of parameters: {model.nparams}')

    if args.dataset == 'val':
        which_loader = val_loader  # NOTE: specify the dataset: train or val?
        which_set = val_dataset
    else:
        which_loader = train_loader
        which_set = train_dataset

    if args.force_dur is not None:
        print(f"Using forced duration from {args.force_dur}")
        utt2dur = dict()
        with open(args.force_dur, 'r') as fr:
            for line in fr.readlines():
                utt2dur[line.strip().split()[0]] = list(map(int, line.strip().split()[1:]))

    met = False
    with torch.no_grad():
        with WriteHelper(f"ark,scp:{os.getcwd()}/{feats_dir}/feats.ark,{feats_dir}/feats.scp") as feats:
            # NOTE: its necessary to add "os.getcwd" here.
            for batch_idx, batch in tqdm(enumerate(which_loader), total=len(which_loader)):
                utts = batch['utt']

                # ============== Loop Controlling block ============
                if met:
                    break
                if args.specify_utt_name is not None:
                    if not utts[0] == args.specify_utt_name:
                        continue
                    else:
                        met = True
                elif batch_idx >= args.max_utt_num:
                    break
                # ==================================================

                x, x_lengths = batch['text_padded'].to(device), batch['input_lengths'].to(device)
                if args.force_dur is not None:
                    force_dur = utt2dur[utts[0]]
                    force_dur = torch.LongTensor(force_dur).unsqueeze(0).to(x.device)
                else:
                    force_dur = None

                # ================== Decode ======================
                if args.use_control_emo:
                    emo = torch.LongTensor([args.control_emo_id]).to(device)
                else:
                    # emo = torch.LongTensor([batch['emo_ids']]).to(device)
                    emo = batch['emo_ids'].to(device)

                if hps.xvector:
                    if args.use_control_spk:
                        xvector = which_set.spk2xvector[args.control_spk_name]
                        xvector = torch.FloatTensor(xvector).squeeze().unsqueeze(0).to(device)
                    else:
                        xvector = batch['xvector'].to(device)

                    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=args.noise,
                                                       stoc=args.stoc, spk=xvector,emo=emo, length_scale=1.,
                                                       classifier_free_guidance=args.guidance,
                                                       force_dur=force_dur)
                else:
                    if args.use_control_spk:
                        sid = torch.LongTensor([args.control_spk_id]).to(device)
                    else:
                        sid = batch['spk_ids'].to(device)

                    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=args.noise,
                                                       stoc=args.stoc, spk=sid, emo=emo, length_scale=1.,
                                                       classifier_free_guidance=args.guidance,
                                                       force_dur=force_dur)
                # =================================================

                # print(y_dec.shape)

                if args.use_control_spk:
                    if args.use_control_emo:
                        save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]-[emo_{args.control_emo_id}]{utts[0]}"
                    else:
                        save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]{utts[0]}"
                else:
                    if args.use_control_emo:
                        save_utt_name = f"[emo_{args.control_emo_id}]{utts[0]}"
                    else:
                        save_utt_name = f"{utts[0]}"

                feats(save_utt_name, y_dec.squeeze().cpu().numpy().T)  # save to ark and scp, mel: (L, 80)


if __name__ == '__main__':
    hps, args = utils.get_hparams_decode()
    ckpt = utils.latest_checkpoint_path(hps.model_dir, "EMA_grad_*.pt")

    if args.use_control_spk:
        feats_dir = f"synthetic_wav/{args.model}/tts_other_spk"
    else:
        feats_dir = f"synthetic_wav/{args.model}/tts_gt_spk"
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)

    evaluate(hps, args, ckpt, feats_dir)
