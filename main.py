import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 点云数据集类
class PointCloudDataset(Dataset):
    def __init__(self, data_root="data", num_points=1024):
        self.data_root = data_root
        self.num_points = num_points
        self.categories = ['bs', 'ea', 'ht', 'lr', 'ra', 'ss']
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        self.data = []
        self.labels = []

        # 加载所有数据
        for cat in self.categories:
            cat_dir = os.path.join(data_root, cat)
            for file in os.listdir(cat_dir):
                if file.endswith('.txt'):
                    points = np.loadtxt(os.path.join(cat_dir, file), delimiter=',')[:, :3]
                    # 随机采样固定数量的点
                    if len(points) >= self.num_points:
                        indices = np.random.choice(len(points), self.num_points, replace=False)
                    else:
                        indices = np.random.choice(len(points), self.num_points, replace=True)
                    points = points[indices]

                    # 中心化
                    centroid = np.mean(points, axis=0)
                    points = points - centroid
                    # 归一化
                    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                    points = points / m

                    self.data.append(points)
                    self.labels.append(self.cat_to_idx[cat])

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return points, label


# Point Transformer Layer
class PointTransformerLayer(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.dim = dim

        # 位置编码MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # 特征转换
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        # 注意力权重生成
        self.linear_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, x, pos):
        batch_size, num_points, _ = x.shape

        # KNN查找邻域点
        inner = torch.matmul(pos, pos.transpose(2, 1))
        xx = torch.sum(pos ** 2, dim=2, keepdim=True)
        dist = xx + xx.transpose(2, 1) - 2 * inner
        idx = dist.topk(k=self.k, dim=-1, largest=False)[1]

        # 获取邻域点的特征和位置
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.contiguous()
        pos = pos.contiguous()

        # (batch_size, num_points, k, dim)
        neighbors = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, self.k, -1)
        pos_neighbors = pos.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, self.k, -1)

        # 相对位置
        pos_relative = pos.unsqueeze(2) - pos_neighbors
        pos_embedding = self.pos_mlp(pos_relative)

        # 计算注意力
        q = self.linear_q(x).unsqueeze(2)  # (batch_size, num_points, 1, dim)
        k = self.linear_k(neighbors)  # (batch_size, num_points, k, dim)
        v = self.linear_v(neighbors)  # (batch_size, num_points, k, dim)

        # 向量注意力
        attn = self.linear_attn(q - k + pos_embedding)  # (batch_size, num_points, k, dim)
        attn = F.softmax(attn, dim=2)

        # 聚合特征
        out = torch.sum(attn * (v + pos_embedding), dim=2)  # (batch_size, num_points, dim)

        return out


# Point Transformer分类网络
class PointTransformerCls(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # 特征提取
        self.linear1 = nn.Linear(3, 32)
        self.transformer1 = PointTransformerLayer(dim=32)
        self.linear2 = nn.Linear(32, 64)
        self.transformer2 = PointTransformerLayer(dim=64)
        self.linear3 = nn.Linear(64, 128)
        self.transformer3 = PointTransformerLayer(dim=128)
        self.linear4 = nn.Linear(128, 256)
        self.transformer4 = PointTransformerLayer(dim=256)

        # 全局特征
        self.linear5 = nn.Linear(256, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.linear6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)

        # 分类头
        self.classifier = nn.Linear(256, num_classes)

        self.dp1 = nn.Dropout(p=0.4)
        self.dp2 = nn.Dropout(p=0.4)

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        pos = x.clone()

        # 特征提取
        x = self.linear1(x)
        x = self.transformer1(x, pos)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.transformer2(x, pos)
        x = F.relu(x)

        x = self.linear3(x)
        x = self.transformer3(x, pos)
        x = F.relu(x)

        x = self.linear4(x)
        x = self.transformer4(x, pos)
        x = F.relu(x)

        # 全局特征
        x = torch.max(x, dim=1)[0]

        x = F.relu(self.bn5(self.linear5(x)))
        x = self.dp1(x)
        x = F.relu(self.bn6(self.linear6(x)))
        x = self.dp2(x)

        # 分类
        x = self.classifier(x)

        return x


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model = model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return model


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    dataset = PointCloudDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 创建模型
    model = PointTransformerCls()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

    print("Training completed!")


if __name__ == "__main__":
    main()