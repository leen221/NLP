"""
=============================================================
  Deep Neural Network for Question Answering (QA) on SQuAD
  Model: BERT fine-tuned for Extractive QA
  Dataset: SQuAD v1.1 (Stanford Question Answering Dataset)
=============================================================
""" 

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  0. Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
#  1. Hyperparameters
# ─────────────────────────────────────────────
CONFIG = {
    "model_name":        "bert-base-uncased",
    "max_length":        384,
    "doc_stride":        128,
    "batch_size":        4,            # ✅ قللنا من 8 لـ 4
    "epochs":            3,
    "learning_rate":     3e-5,
    "warmup_ratio":      0.1,
    "weight_decay":      0.01,
    "max_answer_length": 30,
    "n_best_size":       20,
    "train_samples":     5000,         # ✅ بدل None
    "val_samples":       500,          # ✅ بدل None
    "output_dir":        "outputs",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ─────────────────────────────────────────────
#  2. Dataset Loading
# ─────────────────────────────────────────────
def load_squad():
    print("[INFO] Loading SQuAD v1.1 …")
    squad = load_dataset("squad")
    train_data = squad["train"]
    val_data   = squad["validation"]

    if CONFIG["train_samples"]:
        train_data = train_data.select(range(CONFIG["train_samples"]))
    if CONFIG["val_samples"]:
        val_data   = val_data.select(range(CONFIG["val_samples"]))

    print(f"[INFO] Train examples : {len(train_data)}")
    print(f"[INFO] Val   examples : {len(val_data)}")
    return train_data, val_data


# ─────────────────────────────────────────────
#  3. Tokenisation & Feature Extraction
# ─────────────────────────────────────────────
def prepare_train_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=CONFIG["max_length"],
        truncation="only_second",
        stride=CONFIG["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map      = tokenized.pop("overflow_to_sample_mapping")
    offset_map      = tokenized.pop("offset_mapping")
    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_map):
        input_ids  = tokenized["input_ids"][i]
        cls_index  = input_ids.index(tokenizer.cls_token_id)
        seq_ids    = tokenized.sequence_ids(i)
        sample_idx = sample_map[i]
        answers    = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char   = start_char + len(answers["text"][0])

            ctx_start = 0
            while seq_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(seq_ids) - 1
            while seq_ids[ctx_end] != 1:
                ctx_end -= 1

            if not (offsets[ctx_start][0] <= start_char and
                    offsets[ctx_end][1]   >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                token_start = ctx_start
                while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                    token_start += 1
                start_positions.append(token_start - 1)

                token_end = ctx_end
                while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                    token_end -= 1
                end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized


def prepare_val_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=CONFIG["max_length"],
        truncation="only_second",
        stride=CONFIG["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized["overflow_to_sample_mapping"]
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        seq_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
        tokenized["example_id"].append(examples["id"][sample_map[i]])

    return tokenized


# ─────────────────────────────────────────────
#  4. PyTorch Dataset Wrapper
# ─────────────────────────────────────────────
class SQuADDataset(Dataset):
    def __init__(self, tokenized_data, is_train=True):
        self.data     = tokenized_data
        self.is_train = is_train
        self.keys     = ["input_ids", "attention_mask", "token_type_ids"]
        if is_train:
            self.keys += ["start_positions", "end_positions"]

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(self.data[k][idx]) for k in self.keys}


# ─────────────────────────────────────────────
#  5. Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", unit="batch")

    for batch in pbar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        start_pos      = batch["start_positions"].to(DEVICE)
        end_pos        = batch["end_positions"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_pos,
            end_positions=end_pos,
        )

        loss = outputs.loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"  ↳ Average train loss: {avg_loss:.4f}")
    return avg_loss


# ─────────────────────────────────────────────
#  6. Evaluation & Metric Computation
# ─────────────────────────────────────────────
def postprocess_predictions(examples, features, raw_predictions, tokenizer):
    start_logits, end_logits = raw_predictions

    example_to_features = {}
    for i, feat in enumerate(features):
        eid = feat["example_id"]
        example_to_features.setdefault(eid, []).append(i)

    predictions = {}
    for example in examples:
        eid      = example["id"]
        context  = example["context"]
        feat_ids = example_to_features.get(eid, [])

        valid_answers = []
        for feat_idx in feat_ids:
            offsets = features[feat_idx]["offset_mapping"]
            sl      = start_logits[feat_idx]
            el      = end_logits[feat_idx]

            start_idxs = np.argsort(sl)[-1 : -CONFIG["n_best_size"] - 1 : -1].tolist()
            end_idxs   = np.argsort(el)[-1 : -CONFIG["n_best_size"] - 1 : -1].tolist()

            for s in start_idxs:
                for e in end_idxs:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > CONFIG["max_answer_length"]:
                        continue
                    valid_answers.append({
                        "score": sl[s] + el[e],
                        "text":  context[offsets[s][0]: offsets[e][1]],
                    })

        if valid_answers:
            predictions[eid] = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]["text"]
        else:
            predictions[eid] = ""

    return predictions


def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(model, val_dataset, val_features_raw, val_data_raw, tokenizer):
    model.eval()
    all_start_logits = []
    all_end_logits   = []

    # ✅ num_workers=0 لتقليل الضغط على CPU
    loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"] * 2, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits   = np.concatenate(all_end_logits)

    predictions = postprocess_predictions(
        val_data_raw, val_features_raw,
        (all_start_logits, all_end_logits), tokenizer,
    )

    em_scores, f1_scores = [], []
    for ex in val_data_raw:
        eid  = ex["id"]
        pred = predictions.get(eid, "")
        gold_answers = ex["answers"]["text"]
        em_scores.append(max(compute_exact_match(pred, g) for g in gold_answers))
        f1_scores.append(max(compute_f1(pred, g) for g in gold_answers))

    avg_em = 100.0 * np.mean(em_scores)
    avg_f1 = 100.0 * np.mean(f1_scores)
    print(f"  ↳ Exact Match : {avg_em:.2f}%")
    print(f"  ↳ F1 Score    : {avg_f1:.2f}%")
    return avg_em, avg_f1, predictions


# ─────────────────────────────────────────────
#  7. Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()

    print(f"\n[INFO] Loading tokenizer & model: {CONFIG['model_name']}")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])
    model     = BertForQuestionAnswering.from_pretrained(CONFIG["model_name"])
    model.to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters    : {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    train_data, val_data = load_squad()

    print("[INFO] Tokenizing training data …")
    train_tokenized = train_data.map(
        lambda ex: prepare_train_features(ex, tokenizer),
        batched=True, remove_columns=train_data.column_names,
    )

    print("[INFO] Tokenizing validation data …")
    val_tokenized = val_data.map(
        lambda ex: prepare_val_features(ex, tokenizer),
        batched=True, remove_columns=val_data.column_names,
    )

    val_features_list = [{
        "offset_mapping": val_tokenized["offset_mapping"][i],
        "example_id":     val_tokenized["example_id"][i],
    } for i in range(len(val_tokenized["input_ids"]))]

    train_set = SQuADDataset(train_tokenized, is_train=True)
    val_set   = SQuADDataset(val_tokenized,   is_train=False)

    # ✅ num_workers=0 و pin_memory=False لتقليل الضغط على CPU
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": CONFIG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer    = AdamW(params, lr=CONFIG["learning_rate"])
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "em": [], "f1": []}
    best_f1 = 0.0

    print(f"\n{'='*55}")
    print(f"  Training for {CONFIG['epochs']} epochs on SQuAD v1.1")
    print(f"{'='*55}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        em, f1, preds = evaluate(model, val_set, val_features_list, val_data, tokenizer)
        history["train_loss"].append(train_loss)
        history["em"].append(em)
        history["f1"].append(f1)

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(os.path.join(CONFIG["output_dir"], "best_model"))
            tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "best_model"))
            print(f"  ✓ Saved best model (F1={best_f1:.2f}%)")

        print()

    hist_path = os.path.join(CONFIG["output_dir"], "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*55}")
    print(f"  Training complete in {elapsed:.1f} min")
    print(f"  Best F1 Score : {best_f1:.2f}%")
    print(f"  Results saved → {CONFIG['output_dir']}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()