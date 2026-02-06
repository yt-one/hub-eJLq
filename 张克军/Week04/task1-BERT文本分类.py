import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer


# 使用pandas 读取数据，按行读取，','分割内容和标签
ds = pd.read_csv("./dataset.csv",sep=",")
class_number = len(ds['label'].unique())
print("文本数据集的总数量:", len(ds))
print("内容数量:", len(ds['text']))
print("标签数量:", len(ds['label']))
print("文本类别:",class_number)

#使用LabelEncoder 处理文本分类任务的数据集，将文本标签转换为数字标签
label_encoder = LabelEncoder()
#塞入标签列表，自动根据标签列表生成数字标签
texts = list(ds['text'])
labels = label_encoder.fit_transform(ds['label'])
#使用 train_test_split 将数据集按9:1的比例分割为训练集和测试集
x_train, x_test, train_labels, test_labels  = train_test_split(texts, labels, test_size=0.1, stratify=ds['label'].values)
#使用系统的环境变量 MODEL_PATH 获取模型根路径，以后的所有预训练模型都在这个目录下面
model_base_path = os.environ.get("MODEL_BASE_PATH")
tokenizers_path = model_base_path +'/google-bert/bert-base-chinese'
model_path = model_base_path + '/google-bert/bert-base-chinese'


#使用 transformers的BertTokenizer获取分词器和BertForSequenceClassification获取模型
#分词器，自动加载传入目录下面的，vocab.txt, tokenizer_config.json, special_tokens_map.json, added_tokens.json
tokenizer = BertTokenizer.from_pretrained(tokenizers_path)

#检查是否存在results目录，如果存在则使用最新的checkpoint继续训练
has_result = False
has_result_ignore_tran = False
if os.path.exists('./results') and os.listdir('./results'):
    #获取最新的checkpoint目录
    checkpoints = [d for d in os.listdir('./results') if d.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        model_path = f'./results/{latest_checkpoint}'
        print(f"检测到已有训练结果，使用checkpoint: {model_path}")
        has_result = True

#模型，自动加载传入目录下面的，pytorch_model.bin, config.json, vocab.txt, special_tokens_map.json, added_tokens.json
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=class_number)

#使用分词器对训练集和测试集的文本进行编码
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

#将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
#input_ids 和 attention_mask 是BERT模型的输入字段是固定写法
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 文本的attention mask
    'label': train_labels,                               # 文本的标签 ID
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'label': test_labels
})
print("训练集数量:", len(train_dataset))
print("测试集数量:", len(test_dataset))

#定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

#使用简单的 Trainer 和TrainingArguments训练参数进行训练
train_epochs=4
if has_result:
    train_epochs = 2
training_args = TrainingArguments(
    output_dir='./results',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=train_epochs,  # 训练的总轮数
    per_device_train_batch_size=16,  # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,  # 评估时每个设备的批次大小
    warmup_steps=500,  # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    logging_dir='./logs',  # 日志存储目录
    logging_steps=100,  # 每隔100步记录一次日志
    eval_strategy="epoch",  # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",  # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
)

# 创建 Trainer 对象,简化BERT文本分类训练任务
#这里的参数 ，只要指定训练的数据集，文本标签，就能自动进入训练，自动计算损失，梯度更新
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,
)
if not has_result_ignore_tran:
    # 开始训练
    trainer.train()
    # 在测试集上进行最终评估
    eval_results = trainer.evaluate()
    print("最终评估结果:", eval_results)
    accuracy = round(eval_results['eval_accuracy'] * 100, 2)
    print(f"测试集准确率: {accuracy}%")

#输入文本，预测输出分类
def predict(text):
    # 1. 使用 tokenizer 编码输入文本
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    # 假设 inputs 是你的输入数据字典
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 2. 模型推理
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. 获取预测结果
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    return predicted_label

def test(text):
    res = predict(text)
    print(f"输入文本: {text} -> {res}")

test("你好，很高兴见到你！")
test("我非常想学习深度学习")
test("真是个二傻子，什么都做不好")
test("干死你个狗日的日本鬼子")
test("她到底怎么生气了")
test("你是个什么东西")
test("我终于中了五百万")
test("你到底想干什么")
test("这日子好难啊，以后怎么办啊？")
test("哈哈哈，我终于找到理想工作了。")
"""
最终评估结果: {'eval_loss': 0.06097372621297836, 'eval_accuracy': 0.9855769230769231, 'eval_runtime': 0.8911, 'eval_samples_per_second': 466.836, 'eval_steps_per_second': 29.177, 'epoch': 2.0}
测试集准确率: 98.56%
输入文本: 你好，很高兴见到你！ -> 开心
输入文本: 我非常想学习深度学习 -> 平静
输入文本: 真是个二傻子，什么都做不好 -> 厌恶
输入文本: 干死你个狗日的日本鬼子 -> 厌恶
输入文本: 她到底怎么生气了 -> 疑问
输入文本: 你是个什么东西 -> 疑问
输入文本: 我终于中了五百万 -> 开心
输入文本: 你到底想干什么 -> 疑问
输入文本: 这日子好难啊，以后怎么办啊？ -> 关心
输入文本: 哈哈哈，我终于找到理想工作了。 -> 开心

"""