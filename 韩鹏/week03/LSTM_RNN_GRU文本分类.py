import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 文本处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 最大输入文本长度
max_len = 40

# 自定义数据集-》为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 同意的写法，底层pytorch深度学习/大模型
class CharDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度
    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)
    # 获取单个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 定义RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # batch_size * seq_length -> batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)
        # batch_size * seq_length * embedding_dim -> batch_size * seq_length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # batch_size * hidden_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # batch_size * seq_length -> batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)
        # batch_size * seq_length * embedding_dim -> batch_size * seq_length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # batch_size * hidden_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # batch_size * seq_length -> batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)
        # batch_size * seq_length * embedding_dim -> batch_size * seq_length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # batch_size * hidden_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# 数据封装
dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 设置参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# rnn模型、损失函数、优化器
rnn_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

# gru模型、损失函数、优化器
gru_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# lstm模型、损失函数、优化器
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 模型配置信息
model_configs = [
    {
        "model": rnn_model,
        "criterion": rnn_criterion,
        "optimizer": rnn_optimizer,
        "name": "RNN",
        "loss_list": [],
    },
    {
        "model": gru_model,
        "criterion": gru_criterion,
        "optimizer": gru_optimizer,
        "name": "GRU",
        "loss_list": [],
    },
    {
        "model": lstm_model,
        "criterion": lstm_criterion,
        "optimizer": lstm_optimizer,
        "name": "LSTM",
        "loss_list": [],
    }
]

# 训练模型
num_epochs = 4
for model_config in model_configs:
    name = model_config["name"]
    model = model_config["model"]
    criterion = model_config["criterion"]
    optimizer = model_config["optimizer"]
    print(f"Training {name} model...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        model_config["loss_list"].append(running_loss / len(dataloader))

# 测试模型方法
def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

# 测试
for model_config in model_configs:
    name = model_config["name"]
    model = model_config["model"]
    print(f"Testing {name} model...")

    new_text = "帮我导航到北京"
    predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 绘制模型损失图
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置中文字体（如果系统中有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("使用默认字体")

plt.figure(figsize=(10, 6))
for model_config in model_configs:
    name = model_config["name"]
    model = model_config["model"]
    plt.plot(range(1, num_epochs + 1), model_config["loss_list"], label=name)
plt.title("模型损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

