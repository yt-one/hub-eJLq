import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 生成模拟数据：sin函数
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(500, 1)  # 添加一些噪声

# 转换成torch张量(tensor)
X_tensor = torch.from_numpy(X_numpy).float()
y_tensor = torch.from_numpy(y_numpy).float()

print("数据生成完成！")
print(f"X_tensor 形状: {X_tensor.shape}")
print(f"y_tensor 形状: {y_tensor.shape}")
print("---" * 10)

# 定义多层神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = Net()
print("模型创建完成！")
print(model)
print("---" * 10)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
# 优化器为Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_tensor)

    # 计算损失
    loss = loss_fn(y_pred, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每训练500次打印一次损失函数
    if (epoch + 1) % 500 == 0:
        print(f"Epoch[{epoch + 1} / {num_epochs}], loss: {loss.item():.6f}")

print("\n训练完成！")
print("---" * 10)

# 绘制结果
model.eval()
with torch.no_grad():
    y_predicted = model(X_tensor)
y_predicted_numpy = y_predicted.numpy()

# 绘制原始数据和预测结果
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label="原始数据 (含噪声)", color="blue", alpha=0.3, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label="真实 sin(x)", color="green", linestyle='--', linewidth=2)
plt.plot(X_numpy, y_predicted_numpy, label="神经网络预测", color="red", linewidth=2)
plt.title("多层神经网络拟合 Sin 函数")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


