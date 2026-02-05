import collections
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline
)

# =========================
# 0. 配置
# =========================
MODEL_CKPT = "bert-base-uncased"   # 中文可换成 "bert-base-chinese"
MAX_LENGTH = 384                  # SQuAD 常用 384
DOC_STRIDE = 128                  # 滑窗步长
BATCH_SIZE = 16
EPOCHS = 1                        # 演示先 1；想更好可 2~3
LR = 3e-5

# =========================
# 1) 加载模型 & tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CKPT)

# =========================
# 2) 读取 SQuAD 数据集
# =========================
raw_datasets = load_dataset("squad")  # 包含 train / validation
raw_datasets["train"][0] # keys: ['id','title','context','question','answers']

# =========================
# 3) 预处理：训练集（把答案字符位置 -> token start/end）
# =========================
def prepare_train_features(examples):
    questions = [q.strip() for q in examples["question"]]
    # 注意：BERT QA 标准做法：question 在前，context 在后，截断只截 context
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 一个样本可能被滑窗切成多个 feature
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        # 若该问题无答案（SQuAD v1 基本都有），这里只给 CLS 位置
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # sequence_ids: None / 0(question) / 1(context)
        sequence_ids = tokenized.sequence_ids(i)

        # 找到 context token 的起止范围
        ctx_start = 0
        while sequence_ids[ctx_start] != 1:
            ctx_start += 1
        ctx_end = len(sequence_ids) - 1
        while sequence_ids[ctx_end] != 1:
            ctx_end -= 1

        # 答案不在这个窗口里 -> 用 CLS
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 找 start token
            token_start = ctx_start
            while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            # 找 end token
            token_end = ctx_end
            while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# 应用预处理
train_features = raw_datasets["train"].map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on train dataset",
)

validation_features = raw_datasets["validation"].map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
    desc="Running tokenizer on validation dataset",
)

# ==============================
# 4. 训练配置 & Trainer
# ==============================
args = TrainingArguments(
    output_dir="./bert-squad-finetuned",
    eval_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",  # 关闭 wandb 等
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_features,
    eval_dataset=validation_features,
    data_collator=default_data_collator,
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("./bert-squad-final")
