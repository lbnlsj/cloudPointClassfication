import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
from sklearn.manifold import TSNE

# 导入模型定义
from baseline import PointTransformer
from hdpt import ImprovedPointTransformer, random_point_dropout, random_scale_point_cloud, shift_point_cloud, \
    rotate_point_cloud


class MangroveDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, mode='train', transform=False):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform and mode == 'train'
        self.categories = ['bs', 'ea', 'ht', 'lr', 'ra', 'ss']

        self.points = []
        self.labels = []

        for i, category in enumerate(self.categories):
            category_dir = os.path.join(data_dir, category)
            files = [f for f in os.listdir(category_dir) if f.endswith('.txt')]

            for file in files:
                points = np.loadtxt(os.path.join(category_dir, file), delimiter=',')
                if len(points.shape) == 1:
                    points = points.reshape(1, -1)
                points = self.process_pointcloud(points)
                self.points.append(points)
                self.labels.append(i)

        self.points = np.array(self.points)
        self.labels = np.array(self.labels)

        # 标准化
        self.scaler = StandardScaler()
        points_reshaped = self.points.reshape(-1, 3)
        points_normalized = self.scaler.fit_transform(points_reshaped)
        self.points = points_normalized.reshape(-1, self.num_points, 3)

    def process_pointcloud(self, points):
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
        point_cloud = self.points[idx].copy()
        label = self.labels[idx]

        if self.transform:
            point_cloud = random_point_dropout(point_cloud.reshape(1, -1, 3))
            point_cloud = random_scale_point_cloud(point_cloud)
            point_cloud = shift_point_cloud(point_cloud)
            point_cloud = rotate_point_cloud(point_cloud)
            point_cloud = point_cloud.reshape(self.num_points, 3)

        return torch.FloatTensor(point_cloud), torch.LongTensor([label])[0]


class Experiment:
    def __init__(self, data_dir, model_type='baseline', batch_size=32, num_points=1024,
                 learning_rate=0.001, num_epochs=100):
        self.data_dir = data_dir
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_points = num_points
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        self.exp_dir = f'experiments/{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.exp_dir, exist_ok=True)

        self.setup_data()
        self.setup_model()

    def setup_data(self):
        # 加载完整数据集
        full_dataset = MangroveDataset(self.data_dir, self.num_points,
                                       transform=self.model_type == 'hdpt')

        # 划分数据集
        train_idx, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.3)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.67)

        # 创建数据加载器
        self.train_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, train_idx),
            batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, val_idx),
            batch_size=self.batch_size
        )
        self.test_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, test_idx),
            batch_size=self.batch_size
        )

    def setup_model(self):
        if self.model_type == 'baseline':
            self.model = PointTransformer(num_class=6, num_points=self.num_points)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.model = ImprovedPointTransformer(num_class=6, num_points=self.num_points)
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.learning_rate, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=1e-6
            )

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for points, labels in tqdm(self.train_loader, desc='Training'):
            points, labels = points.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if self.model_type == 'hdpt':
            self.scheduler.step()

        return total_loss / len(self.train_loader), 100. * correct / total

    def evaluate(self, loader):
        self.model.eval()
        predictions = []
        labels_list = []
        features_list = []

        with torch.no_grad():
            for points, labels in loader:
                points, labels = points.to(self.device), labels.to(self.device)
                outputs = self.model(points)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

                # 收集特征用于t-SNE
                if hasattr(self.model, 'mlp'):
                    features = outputs.cpu().numpy()
                    features_list.extend(features)

        return np.array(predictions), np.array(labels_list), np.array(features_list)

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{self.exp_dir}/confusion_matrix_epoch_{epoch}.png')
        plt.close()

    def plot_tsne(self, features, labels, epoch):
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title(f't-SNE Visualization - Epoch {epoch}')
        plt.savefig(f'{self.exp_dir}/tsne_epoch_{epoch}.png')
        plt.close()

    def run(self):
        train_losses = []
        train_accs = []
        val_accs = []
        best_val_acc = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 验证
            val_pred, val_true, val_features = self.evaluate(self.val_loader)
            val_acc = 100. * (val_pred == val_true).mean()
            val_accs.append(val_acc)

            epoch_time = time.time() - start_time

            # 记录结果
            print(f'Epoch [{epoch + 1}/{self.num_epochs}] - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Acc: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, f'{self.exp_dir}/best_model.pth')

                # 生成可视化
                self.plot_confusion_matrix(val_true, val_pred, epoch)
                if len(val_features) > 0:
                    self.plot_tsne(val_features, val_true, epoch)

        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(f'{self.exp_dir}/training_curves.png')
        plt.close()

        # 测试集评估
        self.model.load_state_dict(torch.load(f'{self.exp_dir}/best_model.pth')['model_state_dict'])
        test_pred, test_true, _ = self.evaluate(self.test_loader)
        test_acc = 100. * (test_pred == test_true).mean()

        # 保存分类报告
        report = classification_report(test_true, test_pred,
                                       target_names=['bs', 'ea', 'ht', 'lr', 'ra', 'ss'])
        with open(f'{self.exp_dir}/test_report.txt', 'w') as f:
            f.write(f'Test Accuracy: {test_acc:.2f}%\n\n')
            f.write(report)


def main():
    # 基础模型实验
    exp_baseline = Experiment(
        data_dir='mangrove_txt_result',
        model_type='baseline',
        batch_size=32,
        num_points=1024,
        learning_rate=0.001,
        num_epochs=100
    )
    exp_baseline.run()

    # HDPT模型实验
    exp_hdpt = Experiment(
        data_dir='mangrove_txt_result',
        model_type='hdpt',
        batch_size=32,
        num_points=1024,
        learning_rate=0.001,
        num_epochs=200
    )
    exp_hdpt.run()


if __name__ == '__main__':
    main()