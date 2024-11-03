import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# Data Augmentation
def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def rotate_point_cloud(batch_data):
    B, N, C = batch_data.shape
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(B):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


class MangrovePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, subset='train', transform=True):
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
            print(f"Processing category {category} with {len(files)} files")

            for file in files:
                if file.endswith('.txt'):
                    try:
                        points = np.loadtxt(os.path.join(category_dir, file), delimiter=',')
                        if len(points.shape) == 1:
                            points = points.reshape(1, -1)
                        points = self.process_pointcloud(points)
                        self.points.append(points)
                        self.labels.append(i)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue

        self.points = np.array(self.points)
        self.labels = np.array(self.labels)

        # Normalize point clouds
        self.scaler = StandardScaler()
        points_reshaped = self.points.reshape(-1, 3)
        points_normalized = self.scaler.fit_transform(points_reshaped)
        self.points = points_normalized.reshape(-1, self.num_points, 3)

        print(f"Loaded {len(self.points)} point clouds")

    def process_pointcloud(self, points):
        if len(points) == 0:
            return np.zeros((self.num_points, 3))

        if len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points - len(points))
            extra_points = points[indices]
            points = np.vstack((points, extra_points))
        elif len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]

        return points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point_cloud = self.points[idx]
        label = self.labels[idx]

        if self.transform:
            # Apply data augmentation
            point_cloud = random_point_dropout(point_cloud.reshape(1, -1, 3))
            point_cloud = random_scale_point_cloud(point_cloud)
            point_cloud = shift_point_cloud(point_cloud)
            point_cloud = rotate_point_cloud(point_cloud)
            point_cloud = point_cloud.reshape(self.num_points, 3)

        return torch.FloatTensor(point_cloud), torch.LongTensor([label])[0]


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, xyz):
        B, N, C = x.shape

        # Farthest point sampling
        fps_idx = torch.arange(N // 2, device=x.device)[None, :].repeat(B, 1)
        new_xyz = index_points(xyz, fps_idx)

        # kNN grouping
        dists = square_distance(new_xyz, xyz)
        group_idx = dists.argsort()[:, :, :self.k]
        grouped_points = index_points(x, group_idx)

        # Feature transformation
        grouped_points = grouped_points.view(B * N // 2, self.k, -1)
        new_points = self.relu(self.bn1(self.fc1(grouped_points).transpose(1, 2))).transpose(1, 2)
        new_points = torch.max(new_points, dim=1)[0]
        new_points = new_points.view(B, N // 2, -1)

        return new_points, new_xyz


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, xyz1, xyz2):
        """
        Input:
            x1: (B, N1, C1) features of points to interpolate to
            x2: (B, N2, C2) features of points to interpolate from
            xyz1: (B, N1, 3) coordinates of points to interpolate to
            xyz2: (B, N2, 3) coordinates of points to interpolate from
        """
        B, N1, C1 = x1.shape
        _, N2, C2 = x2.shape

        # Find 3 nearest neighbors
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]

        # Calculate interpolation weights
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        # Interpolate features
        interpolated_feats = torch.sum(index_points(x2, idx) * weight.view(B, N1, 3, 1), dim=2)

        # Transform features
        interpolated_feats = interpolated_feats.transpose(1, 2)
        interpolated_feats = self.relu(self.bn1(self.fc1(interpolated_feats)))
        interpolated_feats = interpolated_feats.transpose(1, 2)

        return interpolated_feats


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
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.k = k
        self.d_model = d_model

    def forward(self, xyz, features):
        B, N, C = features.shape

        # 转换特征维度
        x = self.fc1(features)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)

        # 计算k近邻
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]

        # 获取近邻点
        knn_xyz = index_points(xyz, knn_idx)

        # 位置编码
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)

        # 计算注意力
        q = self.w_qs(x)[:, :, None]
        k = index_points(self.w_ks(x), knn_idx)
        v = index_points(self.w_vs(x), knn_idx)

        attn = self.fc_gamma(q - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(self.d_model), dim=2)

        # 聚合特征
        agg = torch.sum(attn * (v + pos_enc), dim=2)

        # 残差连接和norm
        out = self.fc2(agg)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out + x)

        return out


class ImprovedPointTransformer(nn.Module):
    def __init__(self, num_class=6, num_points=1024):
        super().__init__()
        self.num_points = num_points

        # Encoder layers
        self.transformer1 = PointTransformerLayer(3, 64)
        self.down1 = TransitionDown(64, 128)
        self.transformer2 = PointTransformerLayer(128, 128)
        self.down2 = TransitionDown(128, 256)
        self.transformer3 = PointTransformerLayer(256, 256)
        self.down3 = TransitionDown(256, 512)
        self.transformer4 = PointTransformerLayer(512, 512)

        # Decoder layers
        self.up1 = TransitionUp(512, 256)
        self.transformer5 = PointTransformerLayer(256, 256)
        self.up2 = TransitionUp(256, 128)
        self.transformer6 = PointTransformerLayer(128, 128)
        self.up3 = TransitionUp(128, 64)
        self.transformer7 = PointTransformerLayer(64, 64)

        # Classification head
        self.mlp = nn.Sequential(
            nn.Linear(512 + 256 + 128 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        xyz = x  # (B, N, 3)

        # Encoder path
        f1 = self.transformer1(xyz, x)  # (B, N, 64)
        f2, xyz2 = self.down1(f1, xyz)  # (B, N/2, 128)
        f2 = self.transformer2(xyz2, f2)
        f3, xyz3 = self.down2(f2, xyz2)  # (B, N/4, 256)
        f3 = self.transformer3(xyz3, f3)
        f4, xyz4 = self.down3(f3, xyz3)  # (B, N/8, 512)
        f4 = self.transformer4(xyz4, f4)

        # Decoder path
        up_f3 = self.up1(f3, f4, xyz3, xyz4)  # (B, N/4, 256)
        up_f3 = self.transformer5(xyz3, up_f3)
        up_f2 = self.up2(f2, up_f3, xyz2, xyz3)  # (B, N/2, 128)
        up_f2 = self.transformer6(xyz2, up_f2)
        up_f1 = self.up3(f1, up_f2, xyz, xyz2)  # (B, N, 64)
        up_f1 = self.transformer7(xyz, up_f1)

        # Multi-scale feature fusion
        global_f4 = torch.max(f4, dim=1)[0]  # (B, 512)
        global_f3 = torch.max(f3, dim=1)[0]  # (B, 256)
        global_f2 = torch.max(f2, dim=1)[0]  # (B, 128)
        global_f1 = torch.max(up_f1, dim=1)[0]  # (B, 64)

        # Concatenate global features
        global_feature = torch.cat([global_f4, global_f3, global_f2, global_f1], dim=-1)

        # Classification
        out = self.mlp(global_feature)
        return out


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Update learning rate
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Average Accuracy: {epoch_acc:.2f}%')

        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'hdpt_best.pth')

    return model


def main():
    print("Starting Hierarchical Dual-path Point Transformer training...")

    # Hyperparameters
    data_dir = 'mangrove_txt_result'
    batch_size = 32
    num_points = 1024
    num_classes = 6
    learning_rate = 0.001
    num_epochs = 200

    # Create dataset and data loader
    try:
        dataset = MangrovePointCloudDataset(data_dir, num_points=num_points)
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
        print(f"Successfully created dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ImprovedPointTransformer(num_class=num_classes,
                                     num_points=num_points).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.01)

    # Train model
    try:
        model = train_model(model,
                            train_loader,
                            criterion,
                            optimizer,
                            device,
                            num_epochs)
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        # Save checkpoint in case of error
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'hdpt_checkpoint.pth')
        print("Model checkpoint saved")


if __name__ == '__main__':
    main()
