import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def process_pointcloud(points, num_points):
    if len(points) == 0:
        return np.zeros((num_points, 3))

    if len(points) < num_points:
        indices = np.random.choice(len(points), num_points - len(points))
        extra_points = points[indices]
        points = np.vstack((points, extra_points))
    elif len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]

    return points


class MangrovePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, subset='train', transform=None):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.categories = ['bs', 'ea', 'ht', 'lr', 'ra', 'ss']

        self.points = []
        self.labels = []

        print("Loading point cloud data...")
        for i, category in enumerate(self.categories):
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory {category} not found")
                continue

            files = os.listdir(category_dir)
            files = files[:100]
            print(f"Processing category {category} with {len(files)} files")

            for file in files:
                if file.endswith('.txt'):
                    try:
                        points = np.loadtxt(os.path.join(category_dir, file), delimiter=',')
                        if len(points.shape) == 1:
                            points = points.reshape(1, -1)
                        points = process_pointcloud(points, self.num_points)
                        self.points.append(points)
                        self.labels.append(i)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue

        if len(self.points) == 0:
            raise RuntimeError("No valid point cloud data found!")

        self.points = np.array(self.points)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.points)} point clouds")

        self.scaler = StandardScaler()
        points_reshaped = self.points.reshape(-1, 3)
        points_normalized = self.scaler.fit_transform(points_reshaped)
        self.points = points_normalized.reshape(-1, self.num_points, 3)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point_cloud = self.points[idx]
        label = self.labels[idx]

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return torch.FloatTensor(point_cloud), torch.LongTensor([label])[0]


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class PointTransformerLayer(nn.Module):
    def __init__(self, d_points, d_model, k=16):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.d_model = d_model

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
        """
        B, N, C = features.shape

        # 转换特征维度
        x = self.fc1(features)  # (B, N, d_model)

        # 计算k近邻
        dists = square_distance(xyz, xyz)  # (B, N, N)
        knn_idx = dists.argsort()[:, :, :self.k]  # (B, N, k)

        # 获取近邻点
        knn_xyz = index_points(xyz, knn_idx)  # (B, N, k, 3)

        # 位置编码
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # (B, N, k, d_model)

        # 计算注意力
        q = self.w_qs(x)[:, :, None]  # (B, N, 1, d_model)
        k = index_points(self.w_ks(x), knn_idx)  # (B, N, k, d_model)
        v = index_points(self.w_vs(x), knn_idx)  # (B, N, k, d_model)

        attn = self.fc_gamma(q - k + pos_enc)  # (B, N, k, d_model)
        attn = F.softmax(attn / np.sqrt(self.d_model), dim=2)

        # 聚合特征
        agg = torch.sum(attn * (v + pos_enc), dim=2)  # (B, N, d_model)
        out = self.fc2(agg)  # (B, N, d_model)

        return out


class PointTransformer(nn.Module):
    def __init__(self, num_class=6, num_points=1024):
        super().__init__()
        self.num_points = num_points

        self.transformer1 = PointTransformerLayer(3, 64)
        self.transformer2 = PointTransformerLayer(64, 128)
        self.transformer3 = PointTransformerLayer(128, 256)
        self.transformer4 = PointTransformerLayer(256, 512)

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        # x: (B, N, 3)
        xyz = x

        # 应用transformer层
        x = self.transformer1(xyz, x)  # (B, N, 64)
        x = self.transformer2(xyz, x)  # (B, N, 128)
        x = self.transformer3(xyz, x)  # (B, N, 256)
        x = self.transformer4(xyz, x)  # (B, N, 512)

        # 全局池化
        x = torch.max(x, dim=1)[0]  # (B, 512)

        # MLP分类器
        x = self.mlp(x)  # (B, num_class)

        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (points, labels) in enumerate(train_loader):
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

            if i % 10 == 0:
                print(f'Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def main():
    print("Starting point cloud classification training...")

    # 设置参数
    data_dir = 'mangrove_txt_result'
    batch_size = 32
    num_points = 1024
    num_classes = 6
    learning_rate = 0.001
    num_epochs = 100

    # 创建数据集和数据加载器
    try:
        dataset = MangrovePointCloudDataset(data_dir, num_points=num_points)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Successfully created dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    model = PointTransformer(num_class=num_classes, num_points=num_points).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    try:
        train_model(model, train_loader, criterion, optimizer, device, num_epochs)
        torch.save(model.state_dict(), 'models/baseline.pth')
        print("Training completed and model saved successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        torch.save(model.state_dict(), 'models/baseline.pth')
        print(f"Training interrupted: {e}")
        print("Model checkpoint saved")


if __name__ == '__main__':
    main()