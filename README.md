# -PepRPB-BERT
ode for PepRPB-BERT model in JCIM submission
PepRPB-BERT: A BERT-based Model with Relative Position Bias for Antimicrobial Peptide Identification
This repository contains the official code and supplementary materials for the PepRPB-BERT model, designed for antimicrobial peptide (AMP) identification and related downstream tasks. The model incorporates Relative Position Bias (RPB) to better capture short-range periodic patterns (i±3/i±4) critical for α-helical antimicrobial peptides.
📋 Table of Contents
- Model Overview
- Pre-training Data
- Environment Setup
- File Structure
- Model Training
- Inference for Candidate Peptides
- Pre-trained Weights
- Citation
🔍 Model Overview
PepRPB-BERT is a transformer-based model optimized for short peptide sequence modeling, with the following key configurations:
- Backbone: BERT-like encoder (6 layers, 8 attention heads, d_model=320)
- Position Encoding: Sinusoidal Positional Encoding + Relative Position Bias (RPB)
- Normalization: RMSNorm (native PyTorch 2.4+)
- Feed-Forward Network: SwiGLU
- Pooling: Mean Pooling
Compared to models using Rotary Position Embedding (RoPE, e.g., ESM2), PepRPB-BERT is more suitable for short peptide sequences (≤50 amino acids) and better captures the α-helical periodic patterns critical for AMP identification.
📊 Pre-training Data
The pre-training dataset was constructed from theUniProt Archive (UniParc) database, containing 19.18 million non-redundant peptide sequences (≤50 amino acids).
Raw data is publicly available at: https://www.uniprot.org/uniparc
Note: We do not upload the full pre-training data to GitHub due to size limitations. Users can download the raw data from the official UniParc link and preprocess it using the provided encoding function.
💻 Environment Setup
Install the required dependencies using requirements.txt:
pip install -r requirements.txt
Key dependencies (consistent with the model code):
- Python ≥ 3.8
- PyTorch ≥ 2.4.0 (for native RMSNorm)
- pandas ≥ 1.4.0
- numpy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- pyarrow ≥ 10.0.0 (for parquet data loading)
📂 File Structure
The repository follows a standard, journal-accepted structure for reproducibility:
PepRPB-BERT/
├── README.md               # This file
├── requirements.txt        # Environment dependencies
├── tokenizer.json          # Amino acid tokenizer
├── vocab.txt               # Amino acid vocabulary
│
├── models/                 # Model architecture
│   └── model_bioamp_v1rpb.py  # PepRPB-BERT model definition
│
├── train/                  # Training scripts
│   ├── train_bioamp_rpb.py     # Main pre-training script (RPB model)
│   
│
├── inference/              # eval scripts
│   └── inference_rpb_classifier.py  # Predict candidate peptides
│
└── data/                   # Example data (no full dataset)
    ├── example_candidates.csv  # Example candidate peptides (10-20 samples)
    └── amp_classification.csv  # Data for training classification head (pre-training/test peptides)
🏋️ Model Training
1. Pre-training (RPB Model)
Run the pre-training script for PepRPB-BERT:
cd PepRPB-BERT
nohup python -u train/train_bioamp_rpb.py > logs_bioamp_rpb.log 2>&1 
Key pre-training details:
- Pre-training data: Local parquet file (processed from UniParc)
- Sampling: 19.18 million sequences per epoch (aligned with PepBERT-large)
- Learning rate scheduling: SGDR + Warmup
- Checkpoints: Saved to checkpoints_bioamp_rpb/
- Logs: Saved to logs_bioamp_rpb.log
2. Fine-tuning (Classification Head)
Fine-tune the pre-trained model with a classification head for AMP identification (script not included here; use the classification head defined in inference/inference_rpb_classifier.py).
🔮 Inference for Candidate Peptides
Use the pre-trained PepRPB-BERT model to predict AMP labels for candidate peptides (e.g., candidates_physicochem_filtered.csv).
Step 1: Prepare Input Data
Prepare candidate peptides in CSV format (example: data/example_candidates.csv):
sequence
FRWFFRSRQRRV
KYKRWRWYWNYQRR
RRGFIFRKRDKF
GSWRWWWWRLRRR
ILSFFRRRKWWRP
Step 2: Run Inference
python inference/inference_rpb_classifier.py
Step 3: View Results
Prediction results are saved to data/RPB_AMP_prediction.csv, including:
- sequence: Candidate peptide sequence
- predicted_label: 1 (AMP) / 0 (non-AMP)
- amp_probability: Probability of being an AMP (0-1)

