"""
model_bioamp_v1rpb.py — PepRPB-BERT (full) ablation baseline
================================================
Final model configuration:
  - RMSNorm (native PyTorch 2.4+)
  - RPB (Relative Position Bias)
  - SwiGLU FFN
  - Mean pooling
"""

import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ──────────────────────────────────────────────────────────
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        d_hidden     = int(d_ff * 2 / 3)
        d_hidden     = (d_hidden + 7) // 8 * 8
        self.w_gate  = nn.Linear(d_model, d_hidden, bias=False)
        self.w_value = nn.Linear(d_model, d_hidden, bias=False)
        self.w_out   = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.dropout(
            self.act(self.w_gate(x)) * self.w_value(x)))


# ──────────────────────────────────────────────────────────
# Input Embeddings
# ──────────────────────────────────────────────────────────
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ──────────────────────────────────────────────────────────
# Residual Connection using RMSNorm
# ──────────────────────────────────────────────────────────
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.RMSNorm(features)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ──────────────────────────────────────────────────────────
# RPB: Relative Position Bias
# ──────────────────────────────────────────────────────────
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_len: int = 52) -> None:
        super().__init__()
        self.max_len = max_len
        self.bias    = nn.Embedding(2 * max_len - 1, num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos  = torch.arange(seq_len, device=self.bias.weight.device)
        rel  = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel  = rel.clamp(-(self.max_len - 1), self.max_len - 1)
        rel  = rel + (self.max_len - 1)
        bias = self.bias(rel)               # (L, L, h)
        return bias.permute(2, 0, 1)       # (h, L, L)


# ──────────────────────────────────────────────────────────
# Multi-Head Attention Block
# ──────────────────────────────────────────────────────────
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float,
                 max_len: int = 52) -> None:
        super().__init__()
        assert d_model % h == 0
        self.d_k     = d_model // h
        self.h       = h
        self.w_q     = nn.Linear(d_model, d_model, bias=False)
        self.w_k     = nn.Linear(d_model, d_model, bias=False)
        self.w_v     = nn.Linear(d_model, d_model, bias=False)
        self.w_o     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rpb     = RelativePositionBias(h, max_len=max_len)

    def forward(self, q, k, v, mask):
        B, L, _ = q.shape
        Q = self.w_q(q).view(B, L, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, L, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, L, self.h, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.rpb(L).unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        scores = self.dropout(scores.softmax(dim=-1))
        x = (scores @ V).transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
        return self.w_o(x)


# ──────────────────────────────────────────────────────────
# Encoder Block
# ──────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, features, self_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff        = ff
        self.res       = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.res[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.res[1](x, self.ff)
        return x


# ──────────────────────────────────────────────────────────
# Encoder with RMSNorm
# ──────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm   = nn.RMSNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ──────────────────────────────────────────────────────────
# MLM Head with RMSNorm
# ──────────────────────────────────────────────────────────
class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dense   = nn.Linear(d_model, d_model)
        self.norm    = nn.RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)
        self.act     = nn.SiLU()
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        return self.decoder(self.norm(self.act(self.dense(x))))


# ──────────────────────────────────────────────────────────
# BioAMPBERT with mean pooling
# ──────────────────────────────────────────────────────────
class BioAMPBERT(nn.Module):
    def __init__(self, encoder, src_embed, src_pos, mlm_head):
        super().__init__()
        self.encoder   = encoder
        self.src_embed = src_embed
        self.src_pos   = src_pos
        self.mlm_head  = mlm_head

    def forward(self, src, src_mask):
        x      = self.src_embed(src)
        x      = self.src_pos(x)
        hidden = self.encoder(x, src_mask)
        logits = self.mlm_head(hidden)
        return logits, hidden


# ──────────────────────────────────────────────────────────
# build_transformer — Unified entry point for model construction
# ──────────────────────────────────────────────────────────
def build_transformer(
    src_vocab_size: int,
    src_seq_len:    int   = 50,
    d_model:        int   = 320,
    N:              int   = 6,
    h:              int   = 8,
    dropout:        float = 0.1,
    d_ff:           int   = 1280,
) -> BioAMPBERT:
    max_len   = src_seq_len + 2
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos   = SinusoidalPositionalEncoding(d_model, src_seq_len, dropout)
    blocks    = [
        EncoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout, max_len=max_len),
            SwiGLUFeedForward(d_model, d_ff, dropout),
            dropout)
        for _ in range(N)
    ]
    encoder  = Encoder(d_model, nn.ModuleList(blocks))
    mlm_head = MLMHead(d_model, src_vocab_size)
    model    = BioAMPBERT(encoder, src_embed, src_pos, mlm_head)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
