import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

np.random.seed(42)
x_train = np.random.uniform(-np.pi, np.pi, 1000).reshape(-1, 1)
y_train = np.sin(x_train)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

model = SinNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

x_test = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_test, np.sin(x_test), label='真实 sin(x)', color='blue', linewidth=2)
plt.plot(x_test, y_pred, label='网络拟合', color='red', linestyle='--', linewidth=2)
plt.scatter(x_train, y_train, color='green', s=10, alpha=0.5, label='训练数据')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin函数拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss_history, color='purple', linewidth=1.5)
plt.xlabel('迭代次数')
plt.ylabel('MSE Loss')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('sin函数拟合结果.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n训练完成！')
print(f'最终训练损失: {loss_history[-1]:.6f}')
print('可视化结果已保存为 sin函数拟合结果.png')
