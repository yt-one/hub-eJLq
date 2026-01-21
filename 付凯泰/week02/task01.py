import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
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

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义多层神经网络模型
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MultiLayerClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        # 创建多个隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
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

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses


# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)

# 不同模型配置的实验
model_configs = [
    {"name": "原始模型 (1层隐藏层, 128节点)", "model": SimpleClassifier(vocab_size, 128, output_dim)},
    {"name": "浅层模型 (1层隐藏层, 64节点)", "model": SimpleClassifier(vocab_size, 64, output_dim)},
    {"name": "深层模型 (2层隐藏层, 128-64节点)", "model": MultiLayerClassifier(vocab_size, [128, 64], output_dim)},
    {"name": "更深模型 (3层隐藏层, 256-128-64节点)",
     "model": MultiLayerClassifier(vocab_size, [256, 128, 64], output_dim)},
    {"name": "宽模型 (1层隐藏层, 256节点)", "model": SimpleClassifier(vocab_size, 256, output_dim)},
]

results = {}

print("=" * 50)
print("开始训练不同配置的模型...")
print("=" * 50)

for config in model_configs:
    print(f"\n训练模型: {config['name']}")
    model = config['model']
    losses = train_model(model, dataloader, num_epochs=20)
    results[config['name']] = losses
    print(f"{config['name']} 训练完成，最终Loss: {losses[-1]:.4f}")

# 绘制损失曲线对比图
plt.figure(figsize=(12, 8))
for name, losses in results.items():
    plt.plot(losses, label=name, linewidth=2)

plt.title('不同模型配置的Loss对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 测试每个模型的预测效果
index_to_label = {i: label for label, i in label_to_index.items()}

test_texts = ["帮我导航到北京", "查询明天北京的天气"]

for config in model_configs:
    print(f"\n{config['name']} 预测结果:")
    model = config['model']
    model.eval()
    for text in test_texts:
        with torch.no_grad():
            tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
            tokenized += [0] * (max_len - len(tokenized))

            bow_vector = torch.zeros(vocab_size)
            for index in tokenized:
                if index != 0:
                    bow_vector[index] += 1

            bow_vector = bow_vector.unsqueeze(0)

            output = model(bow_vector)
            _, predicted_index = torch.max(output, 1)
            predicted_index = predicted_index.item()
            predicted_label = index_to_label[predicted_index]

            print(f"  输入 '{text}' 预测为: '{predicted_label}'")
