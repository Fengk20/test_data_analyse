import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel
)
from tqdm import tqdm

# =========================
# Model definition (same as train_v3)
# =========================
class DebertaTextNumericClassifier(nn.Module):
    def __init__(self, model_name, num_numeric_features, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, numeric_feats):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = text_outputs.last_hidden_state[:, 0]

        num_emb = self.numeric_proj(numeric_feats)

        fused = torch.cat([cls_emb, num_emb], dim=1)
        logits = self.classifier(fused)
        return logits


# =========================
# Dataset
# =========================
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, numeric_cols):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.numeric_cols = numeric_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = (
            f"Prompt: {row['prompt']} "
            f"Response A: {row['response_a']} "
            f"Response B: {row['response_b']}"
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        numeric_feats = torch.tensor(
            row[self.numeric_cols].values.astype(np.float32)
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric_feats": numeric_feats,
            "id": row["id"]
        }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="data/submission-1215-1450.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- Load data ----------
    df = pd.read_csv(args.test_csv)

    # 数值特征列（自动推断）
    numeric_cols = [
        c for c in df.columns
        if c not in [
            "id", "model_a", "model_b",
            "prompt", "response_a", "response_b", "label"
        ]
    ]

    # ---------- Load tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # ---------- Load model ----------
    model = DebertaTextNumericClassifier(
        model_name=args.model_dir,
        num_numeric_features=len(numeric_cols),
        num_labels=3
    )

    # state_dict = torch.load(
    #     os.path.join(args.model_dir, "pytorch_model.bin"),
    #     map_location="cpu"
    # )
    # model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---------- Dataset & Loader ----------
    dataset = TestDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        numeric_cols=numeric_cols
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ---------- Prediction ----------
    all_probs = []
    all_ids = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_feats = batch["numeric_feats"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_feats=numeric_feats
            )

            probs = softmax(logits).cpu().numpy()

            all_probs.append(probs)
            all_ids.extend(batch["id"].tolist())

    all_probs = np.vstack(all_probs)

    # ---------- Save ----------
    out_df = pd.DataFrame({
        "id": all_ids,
        "winner_model_a": all_probs[:, 0],
        "winner_model_b": all_probs[:, 1],
        "winner_tie": all_probs[:, 2],
    })

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
