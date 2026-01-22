# 导入 pandas 库用于数据处理，通常用 pd 作为别名
import pandas as pd

# 导入 PyTorch 深度学习框架
import torch

# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 导入 PyTorch 的优化器模块
import torch.optim as optim

# 从 PyTorch 工具模块中导入数据集和数据加载器类
from torch.utils.data import Dataset, DataLoader

# ...（数据加载和预处理部分保持不变，此处省略注释）...

# 从指定路径读取 CSV 数据集，使用制表符作为分隔符，且文件没有表头
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)

# 提取数据集第一列作为文本列表
texts = dataset[0].tolist()

# 提取数据集第二列作为标签列表（字符串形式）
string_labels = dataset[1].tolist()

# 创建标签到索引的映射字典：为所有不重复的标签分配一个唯一的数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}

# 根据映射关系，将字符串标签列表转换为对应的数字标签列表
numerical_labels = [label_to_index[label] for label in string_labels]

# 初始化字符到索引的映射字典，并添加填充符'<pad>'，索引为0
char_to_index = {'<pad>': 0}

# 遍历所有文本，构建字符到索引的完整映射
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 根据字符到索引的映射，创建反向的索引到字符的映射字典
index_to_char = {i: char for char, i in char_to_index.items()}

# 计算词汇表大小（即所有唯一字符的数量，包括填充符）
vocab_size = len(char_to_index)

# 设置每个文本序列的最大长度
max_len = 40


# 定义一个自定义数据集类，用于处理字符级别的词袋表示
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        # 初始化函数：存储文本、标签、字符映射、最大长度和词汇表大小
        self.texts = texts
        # 将数字标签列表转换为 PyTorch 张量，类型为长整型
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        # 调用内部方法创建词袋向量表示
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        # 内部方法：将文本转换为词袋向量
        tokenized_texts = []
        for text in self.texts:
            # 将每个字符转换为其对应的索引，只取前 max_len 个字符
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 如果文本长度不足 max_len，用 0（填充符索引）进行填充
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            # 初始化一个全零向量，长度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:  # 忽略填充符（索引0）
                    bow_vector[index] += 1  # 对应字符的计数加1
            bow_vectors.append(bow_vector)
        # 将所有样本的词袋向量堆叠成一个张量
        return torch.stack(bow_vectors)

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.texts)

    def __getitem__(self, idx):
        # 根据索引返回一个样本：词袋向量和对应的标签
        return self.bow_vectors[idx], self.labels[idx]


# 定义一个简单的神经网络分类器模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):  # 参数：输入维度、隐藏层维度、输出维度
        # 调用父类 nn.Module 的初始化方法
        super(SimpleClassifier, self).__init__()

        # 创建多个隐藏层
        self.layers = nn.ModuleList()
        # 定义第一个全连接层：从输入维度到隐藏层维度
        #self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # 定义 ReLU 激活函数
        #self.relu = nn.ReLU()

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        # 定义第二个全连接层：从隐藏层维度到隐藏层
        for i in range(1,len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))    #0~1
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.3))
        # 定义第n个全连接层：从隐藏层到输出维度
        #self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        # 输出层
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    '''
        def forward(self, x):
        # 定义前向传播过程：依次通过全连接层1、激活函数、全连接层2
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    '''

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


# 创建数据集实例
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
# 创建数据加载器，用于批量加载数据，设置批大小为32，并打乱顺序
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -> 批量数据

# 定义隐藏层维度
#hidden_dims = 128
hidden_dims = [64, 0, 0]  #可调整模型的层数和节点个数
# 输出维度等于类别数量（即标签映射字典的大小）
output_dim = len(label_to_index)
# 创建模型实例
model = SimpleClassifier(vocab_size, hidden_dims, output_dim)  # 维度和精度有什么关系？（注：此问题在注释中提出，未解答）
# 定义损失函数为交叉熵损失（内部已包含 Softmax 激活）
criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
# 定义优化器为随机梯度下降，学习率为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch：将整个数据集迭代训练一次
# batch：将数据集分成小批量进行训练

# 设置训练轮数
num_epochs = 10
# 开始训练循环
for epoch in range(num_epochs):  # 例如：若数据集有12000个样本，批大小为100，则每轮有120个批次
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 初始化累计损失
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空过往梯度
        outputs = model(inputs)  # 前向传播，获取预测输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累计损失
        if idx % 50 == 0:
            # 每50个批次打印一次当前批次损失
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个 epoch 的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

'''

# 定义一个函数，用于对新文本进行分类
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本转换为索引序列，只取前 max_len 个字符，未登录字符用0（填充符）表示
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 若长度不足 max_len，用0填充
    tokenized += [0] * (max_len - len(tokenized))

    # 初始化一个全零的词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:  # 忽略填充符
            bow_vector[index] += 1  # 对应字符计数加1

    # 在第0维增加一个维度，以适应模型输入要求（批处理维度）
    bow_vector = bow_vector.unsqueeze(0)

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        output = model(bow_vector)  # 前向传播，获取预测输出

    # 获取预测类别索引（即输出中最大值的索引）
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 根据索引到标签的映射，获取预测的标签字符串
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 根据标签到索引的映射，创建反向的索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试新文本分类
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
'''
