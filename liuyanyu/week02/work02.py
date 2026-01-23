#2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import torch # 深度学习框架
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt  # 绘图
import torch.nn as nn # 导入 PyTorch 的神经网络模块，包含层、激活函数、损失函数等
import torch.optim as optim # 导入 PyTorch 的优化器模块，用于定义梯度下降等优化方法

# 1. 生成模拟数据
X_numpy =np.random.rand(200, 1) * 2 * np.pi
# 生成y：sin(x) + 高斯噪声（模拟真实数据）
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(200, 1)

# 将 NumPy 数组转换为 PyTorch 的Tensor（张量），因为 PyTorch 的所有计算都基于 Tensor
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleModel, self).__init__()
       self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
model = SimpleModel(1, 32, 16, 1)
print("多层神经网络初始化完成：")
print(model)  # 打印模型结构
print("---" * 10)

# 损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
criterion = torch.nn.MSELoss() # 回归任务
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2000 # 训练轮数（整个数据集训练 10 次）
losses = []  # 记录每轮的平均Loss
for epoch in range(num_epochs):
    # 前向传播：用模型预测y
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)
    losses.append(loss.item())  # 记录loss

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

model.eval()  # 切换到评估模式

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data (sin(x)+noise)', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Model prediction', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('x (0~2π)')
plt.ylabel('y = sin(x)')
plt.title('sin')
plt.legend()
plt.grid(True)
plt.show()
