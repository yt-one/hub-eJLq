#!/usr/bin/env python3
"""
使用生成的动漫数据集进行BERT微调的示例脚本
基于原始的10_BERT文本分类.py进行修改
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# 加载和预处理数据
print("加载动漫数据集...")
dataset_df = pd.read_csv("anime_dataset.csv", sep="\t", header=None, names=['text', 'label'])

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df['label'].values)
texts = list(dataset_df['text'].values)

print(f"数据集大小: {len(texts)} 条记录")
print(f"类别数量: {len(lbl.classes_)} 类")
print(f"类别映射: {dict(zip(lbl.classes_, range(len(lbl.classes_))))}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels,   # 确保训练集和测试集的标签分布一致
    random_state=42    # 固定随机种子保证可重现
)

print(f"训练集大小: {len(x_train)}")
print(f"测试集大小: {len(x_test)}")

# 从预训练模型加载分词器和模型
print("加载BERT模型...")
tokenizer = BertTokenizer.from_pretrained('../../models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('../../models/google-bert/bert-base-chinese', num_labels=3)

# 使用分词器对训练集和测试集的文本进行编码
print("文本编码中...")
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    
    # 计算详细分类报告
    report = classification_report(labels, predictions, 
                                 target_names=lbl.classes_, 
                                 output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': report['macro avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall']
    }
    
    # 添加各类别F1分数
    for i, class_name in enumerate(lbl.classes_):
        metrics[f'f1_{class_name}'] = report[class_name]['f1-score']
    
    return metrics

# 配置训练参数
print("配置训练参数...")
training_args = TrainingArguments(
    output_dir='anime_results',        # 训练输出目录
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=16,      # 训练批次大小
    per_device_eval_batch_size=16,       # 评估批次大小
    warmup_steps=100,                    # 学习率预热步数
    weight_decay=0.01,                   # 权重衰减
    logging_dir='./anime_logs',          # 日志目录
    logging_steps=50,                    # 日志记录间隔
    eval_strategy="steps",               # 评估策略
    eval_steps=100,                      # 评估间隔
    save_strategy="steps",               # 保存策略
    save_steps=100,                      # 保存间隔
    load_best_model_at_end=True,         # 训练结束加载最佳模型
    metric_for_best_model="accuracy",    # 最佳模型评判标准
    greater_is_better=True,              # 越大越好
)

# 实例化 Trainer
print("初始化Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 新增：预测函数
def predict_anime_type(text):
    """预测文本属于哪个动漫类别"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return lbl.inverse_transform([predicted_class])[0]

# 开始训练
print("开始训练...")
trainer.train()

# 在测试集上进行最终评估
print("最终评估...")
final_metrics = trainer.evaluate()
print("最终评估结果:")
for key, value in final_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# 测试新样本
print("\n测试新样本:")
test_samples = [
    "我想看火影忍者",
    "播放蜘蛛侠的电影",
    "查找哪吒之魔童降世的制作背景",
    "续播海贼王到最新话",
    "我想看变形金刚的预告片"
]

print("预测结果:")
for sample in test_samples:
    prediction = predict_anime_type(sample)
    print(f"'{sample}' -> {prediction}")

print("\n训练完成！模型已保存到 ./anime_results/")
print("可以使用 predict_anime_type() 函数对新文本进行分类预测")
