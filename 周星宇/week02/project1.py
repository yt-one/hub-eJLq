import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 数据加载与预处理
# 读取CSV文件
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 提取第一列作为文本列表
string_labels = dataset[1].tolist()  # 提取第二列作为字符串标签列表

# 构建标签映射表：将字符串标签转换为数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将所有的标签转化为对应的数字索引
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符词典：将文本中的每个字符映射到一个数字索引
char_to_index = {'<pad>': 0}  # 0通常用于填充（Padding）
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 建立索引到字符的反向映射
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 词表大小

max_len = 40  # 规定文本的最大长度


# 2. 定义数据集类
class CharBoWDataset(Dataset):
    """
    字符级词袋模型（Bag of Words）数据集
    """
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


# 3. 定义可变深度的分类器模型
class DeepClassifier(nn.Module):
    """
    支持自定义层数和节点个数的全连接神经网络
    """
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(DeepClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 4. 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

# 5. 定义模型配置进行对比测试
# 配置格式：(模型描述, 隐藏层列表)
model_configs = [
    ("浅层窄模型 (1层, 64节点)", [64]),
    ("浅层宽模型 (1层, 256节点)", [256]),
    ("深层窄模型 (3层, [64, 64, 64])", [64, 64, 64]),
    ("深层宽模型 (3层, [256, 128, 64])", [256, 128, 64])
]

results = {}

print("开始对比不同模型结构的训练 Loss 变化...")

for config_name, hidden_layers in model_configs:
    print(f"\n正在训练模型: {config_name}")
    model = DeepClassifier(vocab_size, hidden_layers, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 10
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    results[config_name] = epoch_losses

# 6. 打印对比总结
print("\n" + "="*50)
print("模型 Loss 对比总结 (第1轮 vs 第10轮):")
for name, losses in results.items():
    print(f"{name:30} | Initial Loss: {losses[0]:.4f} | Final Loss: {losses[-1]:.4f} | Reduction: {losses[0]-losses[-1]:.4f}")
print("="*50)

# 7. 推理/预测函数 (使用最后一个训练的模型作为演示)
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)
    _, predicted_index = torch.max(output, 1)
    return index_to_label[predicted_index.item()]

index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n最终模型对 '{new_text}' 的预测结果为: '{predicted_class}'")

new_text2 = "查询明天北京的天气"
predicted_class = classify_text(new_text2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n最终模型对 '{new_text2}' 的预测结果为：'{predicted_class}")
