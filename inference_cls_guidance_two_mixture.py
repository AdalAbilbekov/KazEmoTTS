import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
# from model import GradTTSWithEmo
from model.classifier import SpecClassifier
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
    train_dataset, collate_fn, gradtts_uncond_model = utils.get_correct_class(hps)
    val_dataset, _, _ = utils.get_correct_class(hps, train=False)
    batch_collate = collate_fn
    train_loader = DataLoader(dataset=train_dataset, batch_size=1,
                              collate_fn=batch_collate, drop_last=True,
                              num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4
    val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                            collate_fn=batch_collate, drop_last=True,
                            num_workers=4, shuffle=False)
    gradtts_uncond_model = gradtts_uncond_model(**hps.model).to(device)
    model = SpecClassifier(
        in_dim=hps.data.n_mel_channels,
        d_decoder=hps.model.d_decoder,
        h_decoder=hps.model.h_decoder,
        l_decoder=hps.model.l_decoder,
        k_decoder=hps.model.k_decoder,
        decoder_dropout=hps.model.decoder_dropout,
        n_class=hps.model.n_emos,
        cond_dim=hps.data.n_mel_channels,
        model_type=getattr(hps.model, "classifier_type", "conformer")
    )

    utils.load_checkpoint(f"{os.path.dirname(ckpt)}/grad_uncond.pt", gradtts_uncond_model, None)
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

    met = False
    # with torch.no_grad():
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

            # ================== Decode ======================
            if args.use_control_emo:
                emo1 = torch.LongTensor([args.control_emo_id1]).to(device)
                emo2 = torch.LongTensor([args.control_emo_id2]).to(device)

            else:
                # emo = torch.LongTensor([batch['emo_ids']]).to(device)
                raise NotImplementedError
            if hps.xvector:
                if args.use_control_spk:
                    xvector = which_set.spk2xvector[args.control_spk_name]
                    xvector = torch.FloatTensor(xvector).squeeze().unsqueeze(0).to(device)
                else:
                    xvector = batch['xvector'].to(device)

                y_enc, y_dec, attn = gradtts_uncond_model.classifier_guidance_decode_two_mixture(
                    x, x_lengths,
                    n_timesteps=args.timesteps,
                    temperature=args.noise,
                    stoc=args.stoc,
                    spk=xvector,
                    emo1=emo1,
                    emo2=emo2,
                    emo1_weight=args.emo1_weight,
                    length_scale=1.,
                    classifier_func=model.forward,
                    guidance=args.guidance,
                    classifier_type=model.model_type
                )
            else:
                if args.use_control_spk:
                    sid = torch.LongTensor([args.control_spk_id]).to(device)
                else:
                    sid = batch['spk_ids'].to(device)

                y_enc, y_dec, attn = gradtts_uncond_model.classifier_guidance_decode_two_mixture(
                    x, x_lengths,
                    n_timesteps=args.timesteps,
                    temperature=args.noise,
                    stoc=args.stoc,
                    spk=sid,
                    emo1=emo1,
                    emo2=emo2,
                    emo1_weight=args.emo1_weight,
                    length_scale=1.,
                    classifier_func=model.forward,
                    guidance=args.guidance,
                    classifier_type=model.model_type
                )
            # =================================================

            # print(y_dec.shape)

            if args.use_control_spk:
                if args.use_control_emo:
                    save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]-[emo_{args.control_emo_id}]{utts[0]}"
                else:
                    save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]{utts[0]}"
            else:
                if args.use_control_emo:
                    save_utt_name = f"[emo_{args.control_emo_id1}({args.emo1_weight})_{args.control_emo_id2}({1-args.emo1_weight})]{utts[0]}"
                else:
                    save_utt_name = f"{utts[0]}"

            feats(save_utt_name, y_dec.detach().squeeze().cpu().numpy().T)  # save to ark and scp, mel: (L, 80)


if __name__ == '__main__':
    hps, args = utils.get_hparams_decode_two_mixture()
    ckpt = utils.latest_checkpoint_path(hps.model_dir, "classifier_*.pt")

    # if args.use_control_spk:
    #     feats_dir = f"synthetic_wav/{args.model}/tts_other_spk"
    # else:
    #     feats_dir = f"synthetic_wav/{args.model}/tts_gt_spk"
    feats_dir = f"synthetic_wav/{args.model}/control_emo_mixture"
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)

    evaluate(hps, args, ckpt, feats_dir)
