import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from abc import ABC, abstractmethod
from datasets import Dataset
from sklearn.metrics import accuracy_score, log_loss

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ======================================================
# 1. 基类 Trainer
# ======================================================

class BaseRewardTrainer(ABC):
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None

        # ===== 数值特征列（人工 + 300 维）=====
        self.numeric_cols = (
            [
                "len_diff",
                "len_log_diff",
                "relevance_diff",
                "sim_a_b",
                "code_diff",
                "sent_abs_diff_a_b",
                "sent_diff_prompt_a",
                "sent_diff_prompt_b",
            ]
            + [str(i) for i in range(1, 301)]
        )

    # -------- 分布式辅助 --------
    @property
    def is_main_process(self):
        return int(os.environ.get("LOCAL_RANK", 0)) == 0

    def print_rank0(self, msg):
        if self.is_main_process:
            print(f"[MainProcess] {msg}")

    # -------- WandB --------
    def _init_wandb(self):
        if not self.is_main_process:
            os.environ["WANDB_MODE"] = "disabled"
            return

        if self.config.get("wandb_key"):
            wandb.login(key=self.config["wandb_key"])

        wandb.init(
            project=self.config.get("wandb_project", "human-preference-prediction"),
            name=self.config.get("run_name", "experiment"),
            config=self.config,
            reinit=True,
        )
        self.print_rank0("WandB 初始化完成")

    # -------- 数据加载 --------
    # def load_and_process_data(self):
    #     df = pd.read_csv(self.config["csv_path"])

    #     if self.config.get("debug_mode", False):
    #         self.print_rank0("DEBUG MODE: 使用前 2000 条数据")
    #         df = df.head(20)

    #     # 标签：winner_model_a / b / tie → 0 / 1 / 2
    #     label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    #     df["labels"] = df[label_cols].values.argmax(axis=1)

    #     train_df = df.sample(frac=0.9, random_state=42)
    #     eval_df = df.drop(train_df.index)

    #     train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    #     eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))
    #     return train_ds, eval_ds
    def load_and_process_data(self):
        df = pd.read_csv(self.config["csv_path"])

        if self.config.get("debug_mode", False):
            self.print_rank0("DEBUG MODE: 使用前 2000 条数据")
            df = df.head(57477)

        # ===== 关键修复：标签来源判断 =====
        if "label" in df.columns:
            # 你的真实情况：已经是 0 / 1 / 2
            df["labels"] = df["label"].astype(int)
            self.print_rank0("使用已有的 label 列作为训练标签")
        else:
            # 兼容 one-hot 的老版本
            label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
            missing = [c for c in label_cols if c not in df.columns]
            if len(missing) > 0:
                raise ValueError(
                    f"未找到标签列。需要 'label' 或 {label_cols}，但缺少 {missing}"
                )
            df["labels"] = df[label_cols].values.argmax(axis=1)

        train_df = df.sample(frac=0.9, random_state=42)
        eval_df = df.drop(train_df.index)

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))
        return train_ds, eval_ds
        
    # -------- 评估 --------
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        acc = accuracy_score(labels, preds)
        try:
            ll = log_loss(labels, probs)
        except:
            ll = 0.0
        return {"accuracy": acc, "log_loss": ll}

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def tokenize_function(self, examples):
        pass

    # -------- 主流程 --------
    def run(self):
        self._init_wandb()

        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

        train_ds, eval_ds = self.load_and_process_data()

        train_ds = train_ds.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        eval_ds = eval_ds.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_ds.column_names,
        )

        args = TrainingArguments(
            output_dir=self.config["output_dir"],
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"] * 2,
            gradient_accumulation_steps=self.config.get("grad_accumulation", 1),
            num_train_epochs=self.config["epochs"],
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="log_loss",
            greater_is_better=False,
            fp16=True,
            logging_steps=self.config.get("logging_steps", 50),
            report_to="wandb",
        )
        # args = TrainingArguments(
        #     output_dir=config.output_dir,
        #     per_device_train_batch_size=config.train_batch_size,
        #     per_device_eval_batch_size=config.eval_batch_size,
        #     learning_rate=config.lr,
        #     num_train_epochs=config.epochs,
        #     logging_steps=50,
        #     evaluation_strategy="steps",  # ✅ 在 4.57.3 完全支持
        #     eval_steps=500,
        #     save_steps=500,
        #     report_to="wandb",
        #     remove_unused_columns=False,
        # )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithNumeric(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        self.print_rank0("开始训练")
        trainer.train()

        if self.is_main_process:
            save_path = os.path.join(self.config["output_dir"], "final_model")
            trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            wandb.finish()
            self.print_rank0(f"模型已保存至 {save_path}")


# ======================================================
# 2. 融合模型（DeBERTa + Numeric）
# ======================================================

class DebertaWithNumericFeatures(PreTrainedModel):
    def __init__(self, config, numeric_dim):
        super().__init__(config)

        self.deberta = AutoModel.from_pretrained(config._name_or_path)
        hidden = self.deberta.config.hidden_size

        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, config.num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        numeric_feats=None,
        labels=None,
        **kwargs
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled = outputs.last_hidden_state[:, 0]
        numeric_emb = self.numeric_proj(numeric_feats)

        fused = torch.cat([pooled, numeric_emb], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ======================================================
# 3. DeBERTa Trainer
# ======================================================

class DebertaTrainer(BaseRewardTrainer):
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config["model_name"])

    def get_model(self):
        config = AutoConfig.from_pretrained(
            self.config["model_name"],
            num_labels=3,
            id2label={0: "A", 1: "B", 2: "Tie"},
            label2id={"A": 0, "B": 1, "Tie": 2},
        )

        return DebertaWithNumericFeatures(
            config=config,
            numeric_dim=len(self.numeric_cols),
        )

    def tokenize_function(self, examples):
        sep = self.tokenizer.sep_token
        texts = [
            f"{p} {sep} {ra} {sep} {rb}"
            for p, ra, rb in zip(
                examples["prompt"],
                examples["response_a"],
                examples["response_b"],
            )
        ]

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config["max_length"],
            padding=False,
        )

        numeric_feats = np.stack(
            [examples[col] for col in self.numeric_cols],
            axis=1,
        ).astype(np.float32)

        enc["numeric_feats"] = numeric_feats
        enc["labels"] = examples["labels"]
        return enc


# ======================================================
# 4. 自定义 DataCollator
# ======================================================

class DataCollatorWithNumeric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        numeric_feats = torch.tensor(
            [f.pop("numeric_feats") for f in features],
            dtype=torch.float,
        )
        batch = self.base(features)
        batch["numeric_feats"] = numeric_feats
        return batch


# ======================================================
# 5. 入口
# ======================================================

def train(config):
    trainer = DebertaTrainer(config)
    trainer.run()


if __name__ == "__main__":
    DEBUG = True

    config = {
        "model_name": "microsoft/deberta-v3-xsmall",
        "csv_path": "/root/program/course/25aut_data_analyze/project/dataset_train.csv",
        "output_dir": "./model/deberta_numeric",
        "run_name": "debug-deberta-numeric",
        "wandb_project": "human-preference-prediction",
        "wandb_key": None,
        "max_length": 512,
        "batch_size": 32 if DEBUG else 4,
        "epochs": 1 if DEBUG else 3,
        "learning_rate": 1e-5,
        "debug_mode": DEBUG,
        "logging_steps": 10,
    }

    train(config)
