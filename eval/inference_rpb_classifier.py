"""
Inference / Prediction Script for PepRPB-BERT
Classify candidate peptides using the fine-tuned RPB model
Function: Load pre-trained RPB model -> predict AMP labels for candidate peptides
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ====================== 固定配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 50
VOCAB_SIZE = 29
D_MODEL = 320
BATCH_SIZE = 16
PAD_ID = 1
DROPOUT = 0.3

# 氨基酸编码表
AA_TO_IDX = {
    'L':5,'A':6,'S':7,'R':8,'V':9,'G':10,'K':11,'I':12,
    'E':13,'T':14,'F':15,'P':16,'D':17,'M':18,'N':19,
    'Q':20,'Y':21,'H':22,'C':23,'W':24,
    'X':25,'B':26,'Z':27,'U':28,
}

# ====================== 分类头 ======================
class ClassificationHead(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_model, 2)  # binary classification (AMP/non-AMP)

    def forward(self, x):
        return self.fc(self.drop(x))

# ====================== 序列编码 ======================
def encode_sequence(seq):
    ids = [AA_TO_IDX.get(c, 0) for c in str(seq).upper()[:SEQ_LEN]]
    ids += [PAD_ID] * (SEQ_LEN - len(ids))
    return ids

# ====================== 加载 RPB 模型 ======================
def load_rpb_model(ckpt_path):
    from models.model_bioamp_v1rpb import build_transformer

    model = build_transformer(
        src_vocab_size=VOCAB_SIZE,
        src_seq_len=SEQ_LEN,
        d_model=D_MODEL,
        N=6, h=8, dropout=0.0, d_ff=1280
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model

# ====================== 提取特征 + 预测 ======================
def predict_candidates(model, classifier, sequences):
    predictions = []
    probabilities = []

    with torch.no_grad():
        for seq in sequences:
            # Encode
            ids = encode_sequence(seq)
            ids_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
            attn_mask = (ids_tensor != PAD_ID).float()
            mask = attn_mask.unsqueeze(1).unsqueeze(2)

            # Forward
            _, hidden = model(ids_tensor, mask)

            # Mean pooling
            valid = attn_mask.unsqueeze(-1)
            rep = (hidden[0] * valid).sum(0) / valid.sum()
            rep = rep.unsqueeze(0)

            # Predict
            logits = classifier(rep)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            pred = 1 if prob >= 0.5 else 0

            probabilities.append(round(prob, 4))
            predictions.append(pred)

    return predictions, probabilities

# ====================== 主推理函数 ======================
def run_rpb_inference(
    model_ckpt="./checkpoints/best_model.pt",
    classifier_ckpt="./checkpoints/classifier_head.pt",
    input_csv="./data/candidate_peptides.csv",
    output_csv="./data/rpb_prediction_result.csv"
):
    print("🔹 Loading PepRPB-BERT model...")
    model = load_rpb_model(model_ckpt)

    print("🔹 Loading classification head...")
    classifier = ClassificationHead().to(DEVICE)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location=DEVICE))
    classifier.eval()

    print("🔹 Loading candidate peptides...")
    df = pd.read_csv(input_csv)
    sequences = df["sequence"].astype(str).str.upper().tolist()

    print("🔹 Running prediction...")
    preds, probs = predict_candidates(model, classifier, sequences)

    df["predicted_label"] = preds
    df["amp_probability"] = probs
    df.to_csv(output_csv, index=False)

    print(f"✅ Prediction finished! Results saved to: {output_csv}")
    print(f"Total candidates: {len(sequences)}")
    print(f"Predicted AMPs: {sum(preds)}")

# ====================== 运行入口 ======================
if __name__ == "__main__":
    run_rpb_inference(
        model_ckpt="./checkpoints/best_model.pt",
        classifier_ckpt="./checkpoints/classifier_head.pt",
        input_csv="./data/example_candidates.csv",
        output_csv="./data/RPB_AMP_prediction.csv"
    )
