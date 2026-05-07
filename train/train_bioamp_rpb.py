"""
train_bioamp_rpb.py  —  BioAMP-BERT v1+RPB Pre-training Script
=====================================================
Improvements (compared to original train_bioamp.py):
  1. Use local parquet data source (no streaming from HuggingFace)
  2. Randomly sample 1M sequences per epoch (consistent with MSPepBERT/train_bioamp_v2)
  3. SGDR + Warmup learning rate scheduling (stabilizes loss oscillation)
  4. Use model_bioamp_v1rpb.py (v1 + RPB module)

Checkpoints saved in checkpoints_bioamp_rpb/
Log file saved in logs_bioamp_rpb.log

Run command:
  cd ~/PepBERT
  nohup python -u train_bioamp_rpb.py > logs_bioamp_rpb.log 2>&1 &
"""

import os, sys, math, random, time, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR   = os.path.expanduser('~/PepBERT')
sys.path.insert(0, os.path.join(BASE_DIR, 'models'))
from model_bioamp_v1rpb import build_transformer
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'peptide_UniPrac_0_50.parquet')
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'cached_ids_rpb.npy')
TOK_PATH   = os.path.join(BASE_DIR, 'tokenizer.json')
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints_bioamp_rpb')
LOG_PATH   = os.path.join(BASE_DIR, 'logs_bioamp_rpb_full.log')
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────
VOCAB_SIZE   = 29
D_MODEL      = 320
N_LAYERS     = 6
N_HEADS      = 8
D_FF         = 1280
SEQ_LEN      = 50
DROPOUT      = 0.1
BATCH_SIZE   = 256
EPOCHS       = 60  # Resume from epoch20, run additional 20 epochs to align with PepBERT
BASE_LR      = 3e-4
MIN_LR       = 1e-6
WARMUP_STEPS = 1000
MASK_RATE    = 0.15
GRAD_CLIP    = 1.0
MAX_SAMPLES  = 19_180_037   # Full dataset training, aligned with PepBERT-large-UniParc
PAD_ID       = 1
SOS_ID       = 2
EOS_ID       = 3
MASK_ID      = 2           # v1 uses [SOS]=2 as MASK token (consistent with original)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Amino acid vocabulary for v1 (consistent with train_bioamp.py, no tokenizer.json)
AA_TO_IDX = {
    'L':5,'A':6,'S':7,'R':8,'V':9,'G':10,'K':11,'I':12,
    'E':13,'T':14,'F':15,'P':16,'D':17,'M':18,'N':19,
    'Q':20,'Y':21,'H':22,'C':23,'W':24,
    'X':25,'B':26,'Z':27,'U':28,
}

logging.basicConfig(
    level=logging.INFO, format='%(message)s',
    handlers=[logging.FileHandler(LOG_PATH,'a'), logging.StreamHandler()])

def log(msg):
    logging.info(msg)


# ══════════════════════════════════════════════════
# Random Masking (consistent with original v1, no span masking)
# ══════════════════════════════════════════════════
def random_mask(ids, vocab_size=VOCAB_SIZE, mask_id=MASK_ID,
                pad_id=PAD_ID, mask_rate=MASK_RATE):
    ids    = ids.copy()
    labels = np.full_like(ids, -100)
    prob   = np.random.rand(len(ids))
    valid  = (ids != pad_id)
    selected = valid & (prob < mask_rate)
    labels[selected] = ids[selected]
    r = np.random.rand(selected.sum())
    pos = np.where(selected)[0]
    for i, p in zip(pos, r):
        if p < 0.80:
            ids[i] = mask_id
        elif p < 0.90:
            ids[i] = random.randint(5, vocab_size - 1)
    return ids, labels


# ══════════════════════════════════════════════════
# Dataset Class
# ══════════════════════════════════════════════════
class PeptideDataset(Dataset):
    def __init__(self, data, max_samples=None):
        if max_samples and max_samples < len(data):
            idx = np.random.permutation(len(data))[:max_samples]
            self.data = data[idx]
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        masked, labels = random_mask(ids)
        attn = (ids != PAD_ID).astype(np.int64)
        return (torch.tensor(masked, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(attn,   dtype=torch.long))


def encode_seq_v1(seq):
    """v1 encoding: direct AA_TO_IDX mapping, no [SOS]/[EOS], pad to SEQ_LEN"""
    ids = [AA_TO_IDX.get(c, 0) for c in str(seq).upper()[:SEQ_LEN]]
    ids += [PAD_ID] * (SEQ_LEN - len(ids))
    return np.array(ids, dtype=np.int64)


def load_or_build_cache():
    if os.path.exists(CACHE_PATH):
        log(f'✅ Loaded cache: {CACHE_PATH}')
        return np.load(CACHE_PATH)
    log('🔄 First run, preprocessing and encoding sequences...')
    import pandas as pd
    df  = pd.read_parquet(DATA_PATH, engine='pyarrow', columns=['Sequence'])
    ids = np.stack([encode_seq_v1(s) for s in df['Sequence']], axis=0)
    np.save(CACHE_PATH, ids)
    log(f'✅ Cache saved: {CACHE_PATH}  ({len(ids):,} sequences)')
    return ids


# ══════════════════════════════════════════════════
# SGDR + Warmup Scheduler (consistent with train_bioamp_v2)
# ══════════════════════════════════════════════════
class WarmupSGDR:
    def __init__(self, opt, warmup_steps, T_0, base_lr, min_lr):
        self.opt          = opt
        self.warmup_steps = warmup_steps
        self.T_0          = T_0
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.step_num     = 0
        self.T_cur        = 0
        self.T_i          = T_0

    def _cosine(self):
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + math.cos(math.pi * self.T_cur / self.T_i))

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
            lr = self._cosine()
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'opt'}

    def load_state_dict(self, d):
        self.__dict__.update(d)


# ══════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════
def main():
    log(f'\n{"="*60}')
    log(f'BioAMP-BERT v1+RPB | RoPE+RMSNorm+SwiGLU+RPB | mean pooling')
    log(f'SGDR+Warmup | Random Masking | Full 19.18M sequences/epoch, aligned with PepBERT-large')
    log(f'Device: {DEVICE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}')
    log(f'{"="*60}\n')

    all_ids = load_or_build_cache()
    log(f'Total data: {len(all_ids):,} sequences  |  Sampled per epoch: {MAX_SAMPLES:,} sequences')

    spe = MAX_SAMPLES // BATCH_SIZE   # Estimated steps per epoch
    log(f'Steps/epoch (estimated): {spe:,}')

    model = build_transformer(
        src_vocab_size=VOCAB_SIZE, src_seq_len=SEQ_LEN,
        d_model=D_MODEL, N=N_LAYERS, h=N_HEADS,
        dropout=DROPOUT, d_ff=D_FF).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'Parameters: {n_params:,}  (v1 backbone + RPB with {8*(2*52-1)} extra params)\n')

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                             betas=(0.9,0.98), weight_decay=0.01, eps=1e-8)
    sch = WarmupSGDR(opt, WARMUP_STEPS, T_0=spe,
                      base_lr=BASE_LR, min_lr=MIN_LR)
    scaler  = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_loss= float('inf')
    start_ep = 1

    latest = os.path.join(CKPT_DIR, 'latest_model.pt')
    if os.path.exists(latest):
        log(f'🔄 Resuming from checkpoint: {latest}')
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        if 'scheduler' in ck:
            sch.load_state_dict(ck['scheduler'])
        start_ep  = ck.get('epoch', 0) + 1
        best_loss = ck.get('best_loss', float('inf'))
        log(f'  Resuming from epoch {start_ep}  best_loss={best_loss:.4f}')

    for epoch in range(start_ep, EPOCHS + 1):
        model.train()
        total, nan_n = 0.0, 0
        t0 = time.time()

        dataset = PeptideDataset(all_ids, max_samples=MAX_SAMPLES)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
        actual_spe = len(loader)
        log(f'  Epoch {epoch}: Sampled {len(dataset):,} sequences  Steps: {actual_spe:,}')

        for step, (src, labels, attn) in enumerate(loader, 1):
            src, labels, attn = src.to(DEVICE), labels.to(DEVICE), attn.to(DEVICE)
            # v1 mask shape: (B, 1, 1, L)
            mask = attn.unsqueeze(1).unsqueeze(2)

            with torch.amp.autocast('cuda'):
                logits, _ = model(src, mask)
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE),
                               labels.reshape(-1))

            if torch.isnan(loss):
                nan_n += 1; opt.zero_grad(); continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            lr = sch.step()
            total += loss.item()

            if step % 500 == 0:
                avg = total / step
                el  = (time.time()-t0)/60
                rm  = el/step*(actual_spe-step)
                log(f'  [rpb|ep{epoch}|step{step:07d}] '
                    f'MLM={avg:.4f}  LR={lr:.2e}  NaN={nan_n}  '
                    f'~{rm:.0f}min remaining')

        ep_loss = total / actual_spe
        el = (time.time()-t0)/60
        log(f'\n[rpb] Epoch {epoch:02d}/{EPOCHS} | '
            f'MLM={ep_loss:.4f}  LR={lr:.2e}  '
            f'time={el:.1f}min  NaN={nan_n}')

        ck = dict(epoch=epoch, model=model.state_dict(),
                  optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                  scheduler=sch.state_dict(),
                  loss=ep_loss, best_loss=best_loss)
        torch.save(ck, latest)
        log('  💾 latest_model.pt updated')
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(ck, os.path.join(CKPT_DIR, 'best_model.pt'))
            log(f'  ✅ best_model saved (loss={best_loss:.4f})')

    log('\n✅ Pre-training completed.')


if __name__ == '__main__':
    main()
