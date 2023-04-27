import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data_collate
import data_loader
from utils import plot_tensor, save_plot
from model.utils import fix_len_compatibility
from model.classifier import SpecClassifier
from text.symbols import symbols
from model.utils import sequence_mask, generate_path
import utils


def feed_model(model, gradtts_uncond_model, x, x_lengths, y, y_lengths, durs, emo_label, spk, criterion=torch.nn.CrossEntropyLoss()):
    with torch.no_grad():
        x, x_lengths, y, y_lengths = gradtts_uncond_model.relocate_input([x, x_lengths, y, y_lengths])  # y: B, 80, L

        spk = gradtts_uncond_model.spk_emb(spk)
        # assert emo_label.sum() == 0, "Check emotion label. It must be all zero."
        emo = gradtts_uncond_model.emo_emb(torch.zeros_like(emo_label).long())  # [B, D]. NOTE: zeros like.
        spk = gradtts_uncond_model.merge_spk_emo(torch.cat([spk, emo], dim=-1))  # [B, D]

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = gradtts_uncond_model.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        attn = generate_path(durs, attn_mask.squeeze(1)).detach()
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  # here mu_x is not cut.
        # mu_y: [B, L, 80]
        mu_y = mu_y.transpose(1, 2)
        # diff_loss, yt = gradtts_uncond_model.decoder.compute_loss(y, y_mask, mu_y, spk)
        t = torch.rand(y.shape[0], dtype=y.dtype, device=y.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0 - 1e-5)
        yt, z = gradtts_uncond_model.decoder.forward_diffusion(y, y_mask, mu_y, t)  # z is sampled from N(0, I)
    mu_y = mu_y.transpose(1, 2)
    classification_logits = model.forward(yt.transpose(1, 2), mu_y, (y_mask == 1.).squeeze(1), t=t)  # [B, C]
    # print(classification_logits.shape)

    if model.model_type == 'CNN' or model.model_type == 'CNN-with-time':
        num_classes = classification_logits.shape[-1]
        cnn_seq_len = classification_logits.shape[1]
        classification_logits = classification_logits.view(-1, num_classes)  # [B x seq_len, C]
        emo_label = emo_label.repeat_interleave(cnn_seq_len)
        t = t.repeat_interleave(cnn_seq_len)

    classification_loss = criterion(classification_logits, emo_label)
    return classification_logits, classification_loss, t


if __name__ == "__main__":
    hps = utils.get_hparams()
    logger_text = utils.get_logger(hps.model_dir)
    logger_text.info(hps)

    out_size = fix_len_compatibility(2 * hps.data.sampling_rate // hps.data.hop_length)  # NOTE: 2-sec of mel-spec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps.train.seed)
    np.random.seed(hps.train.seed)

    print('Initializing logger...')
    log_dir = hps.model_dir
    logger = SummaryWriter(log_dir=log_dir)

    train_dataset, collate, gradtts_uncond_model = utils.get_correct_class(hps)
    test_dataset, _, _ = utils.get_correct_class(hps, train=False)

    print('Initializing data loaders...')

    batch_collate = collate
    loader = DataLoader(dataset=train_dataset, batch_size=hps.train.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4

    print('Initializing model...')
    gradtts_uncond_model = gradtts_uncond_model(**hps.model).to(device)
    gradtts_uncond_model.eval()

    model = SpecClassifier(
        in_dim=hps.data.n_mel_channels,
        d_decoder=hps.model.d_decoder,
        h_decoder=hps.model.h_decoder,
        l_decoder=hps.model.l_decoder,
        k_decoder=hps.model.k_decoder,
        decoder_dropout=hps.model.decoder_dropout,
        n_class=hps.model.n_emos,
        cond_dim=hps.data.n_mel_channels,
        model_type=getattr(hps.model, "classifier_type", 'conformer')
    ).to(device)

    # print('Number of decoder parameters: %.2fm' % (model.decoder.nparams / 1e6))
    print('Total parameters: %.2fm' % (model.nparams / 1e6))

    # use_gt_dur = getattr(hps.train, "use_gt_dur", False)
    # if use_gt_dur:
    #     print("++++++++++++++> Using ground truth duration for training")

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=hps.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['mel']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    gradtts_uncond_model, *_ = utils.load_checkpoint(utils.latest_checkpoint_path("logs/EmotionDataNewSplit_gt_dur_uncond/", "EMA_grad_*.pt"), gradtts_uncond_model, None)
    utils.save_checkpoint(gradtts_uncond_model, optimizer, None, None, checkpoint_path=f"{log_dir}/grad_uncond.pt")

    try:
        model, optimizer, learning_rate, epoch_logged = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "classifier_*.pt"), model, optimizer) # TODO: the modify the load_checkpoint from glowtts
        epoch_start = epoch_logged + 1
        print(f"Loaded checkpoint from {epoch_logged} epoch, resuming training.")
        # optimizer.step_num = (epoch_str - 1) * len(train_dataset)
        # optimizer._update_learning_rate()
        global_step = epoch_logged * len(train_dataset)
    except:
        print(f"Cannot find trained checkpoint, begin to train from scratch")
        epoch_start = 1
        global_step = 0
        learning_rate = hps.train.learning_rate

    criterion = torch.nn.CrossEntropyLoss()
    print('Start training...')
    iteration = global_step
    for epoch in range(epoch_start, hps.train.n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset) // hps.train.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['text_padded'].to(device), \
                               batch['input_lengths'].to(device)
                y, y_lengths = batch['mel_padded'].to(device), \
                               batch['output_lengths'].to(device)
                durs = batch['dur_padded'].to(device)
                if hps.xvector:
                    spk = batch['xvector'].to(device)
                else:
                    spk = batch['spk_ids'].to(torch.long).to(device)
                emo_label = batch['emo_ids'].to(torch.long).to(device)  # [B, ]
                if y_lengths.max() < 180:
                    print(y_lengths)
                    continue

                # dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                #                                                      y, y_lengths,
                #                                                      spk=spk,
                #                                                      emo=emo,
                #                                                      out_size=out_size,
                #                                                      use_gt_dur=use_gt_dur,
                #                                                      durs=batch['dur_padded'].to(device) if use_gt_dur else None)

                # ====== begin training procedures. Copied from GradTTS.compute_loss
                classification_logits, classification_loss, t = feed_model(model, gradtts_uncond_model,
                                                                        x, x_lengths, y, y_lengths, durs, emo_label, spk, criterion)
                # print(classification_logits)
                # ========================================

                loss = classification_loss
                # loss = sum([dur_loss, diff_loss])
                loss.backward()

                # enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                #                                                max_norm=1)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                               max_norm=1)
                optimizer.step()

                # logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                #                   global_step=iteration)
                logger.add_scalar('training/grad_norm',grad_norm,
                                  global_step=iteration)

                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | CE loss : {classification_loss.item()}'
                    # logger_text.info(msg)
                    progress_bar.set_description(msg)

                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, float(np.mean(dur_losses)))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % hps.train.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                # print(item)
                x = item['phn_ids'].to(torch.long).unsqueeze(0).to(device)
                y = item['mel'].unsqueeze(0).to(device)
                if not hps.xvector:
                    spk = item['spk_ids']
                    spk = torch.LongTensor([spk]).to(device)
                else:
                    spk = item["xvector"]
                    spk = spk.unsqueeze(0).to(device)
                emo_label = item['emo_ids']
                emo_label = torch.LongTensor([emo_label]).to(device)
                durs = item['dur'].unsqueeze(0).to(device)

                # emo = emo.unsqueeze(0).to(device)

                x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                y_lengths = torch.LongTensor([y.shape[-1]]).to(device)
                if y_lengths.max() < 180:
                    continue

                classification_logits, classification_loss, t = feed_model(model, gradtts_uncond_model, x, x_lengths,
                                                                        y, y_lengths, durs, emo_label, spk, criterion)
                # (time, 1/0). 1 for correct classification. 0 for wrong.
                classification_pred = classification_logits.argmax(1).cpu().numpy().tolist()
                t = t.cpu().numpy().tolist()
                emo_label = emo_label.repeat_interleave(classification_logits.shape[0] // len(emo_label))
                classification_gt = emo_label.cpu().numpy().tolist()
                # print(classification_gt, classification_pred)
                record = [1 if classification_pred[i] == classification_gt[i] else 0 for i in range(len(classification_pred))]
                for i in range(len(classification_pred)):
                    print(f"Epoch {epoch}: At time {t[i]}, Classification {record[i]}")

        ckpt = model.state_dict()
        # torch.save(ckpt, f=f"{log_dir}/classifier_{epoch}.pt")
        utils.save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path=f"{log_dir}/classifier_{epoch}.pt")
