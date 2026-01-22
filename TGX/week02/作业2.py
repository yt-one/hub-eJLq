import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)  # 生成更多的点以更好地拟合sin函数
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 加入少量噪声使问题更现实

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print(f"输入范围: [{X_numpy.min():.2f}, {X_numpy.max():.2f}]")
print(f"输出范围: [{y_numpy.min():.2f}, {y_numpy.max():.2f}]")
print("---" * 10)

# 2. 定义多层神经网络
class SinNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SinNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

# 创建网络实例
model = SinNet(input_dim=1, hidden_dim1=64, hidden_dim2=32, output_dim=1)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss() # 回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每20个 epoch 打印一次损失
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 5. 生成用于可视化的密集数据点
X_plot = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_true_plot = np.sin(X_plot)
X_plot_tensor = torch.from_numpy(X_plot).float()

# 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
    y_pred_plot = model(X_plot_tensor).numpy()


# 6. 绘制结果
plt.figure(figsize=(12, 8))
plt.plot(X_plot, y_true_plot, label='True sin(x)', color='blue', linewidth=2)
plt.plot(X_plot, y_pred_plot, label='Fitted curve', color='red', linewidth=2, linestyle='--')
plt.scatter(X_numpy[::20], y_numpy[::20], label='Training data', color='lightcoral', alpha=0.5)  # 每20个点取1个以避免过于密集
plt.xlabel('X')
plt.ylabel('y')
plt.title('Neural Network Fitting of sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
