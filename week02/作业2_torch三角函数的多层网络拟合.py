import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#1. 生成模拟数据
X_numpy = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.randn(100,1) * 0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据生成完成")
print("---" * 10)

#2. 定义神经网络
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.hidden1 = nn.Linear(1, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.out(x)
        return x

model = SinNet()
print(model)
print("---" * 10)

#3. 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

#4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        print(f"epoch:{epoch+1}/{num_epochs}, loss:{loss.item():.6f}")
print("\n训练完成")

#5. 绘制结果
model.eval()

with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(12, 6))
plt.plot(X_numpy, y_numpy, 'ro-', label='Noisy Data', markersize=4)
plt.plot(X_numpy, y_predicted, 'b-', label='Fitted Curve', linewidth=2)
plt.legend()
plt.show()
