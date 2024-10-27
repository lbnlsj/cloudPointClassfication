import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import time
import argparse
from tqdm import tqdm


class PointCloudNet(nn.Module):
    def __init__(self, num_classes):
        super(PointCloudNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    points = data[:, :3]
    labels = data[:, -1]
    return torch.FloatTensor(points), torch.LongTensor(labels)


def preprocess_data(points, labels):
    mean = torch.mean(points, dim=0)
    std = torch.std(points, dim=0)
    normalized_points = (points - mean) / std
    return normalized_points, labels


def data_augmentation(points, labels):
    augmented_points = points.clone()
    augmented_labels = labels.clone()

    noise = torch.randn_like(points) * 0.02
    augmented_points += noise

    rotation = torch.rand(1) * 2 * np.pi
    cos, sin = torch.cos(rotation), torch.sin(rotation)
    rotation_matrix = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    augmented_points = torch.matmul(augmented_points, rotation_matrix)

    return torch.cat([points, augmented_points]), torch.cat([labels, augmented_labels])


def prepare_dataloader(points, labels, batch_size):
    dataset = TensorDataset(points, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def visualize_results(train_losses, val_losses, all_preds, all_labels, class_names):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    points, labels = load_data(args.data_path)
    points, labels = preprocess_data(points, labels)
    points, labels = data_augmentation(points, labels)

    num_classes = len(torch.unique(labels))
    class_names = [f"Class {i}" for i in range(num_classes)]

    split1 = int(0.7 * len(points))
    split2 = int(0.85 * len(points))
    train_points, train_labels = points[:split1], labels[:split1]
    val_points, val_labels = points[split1:split2], labels[split1:split2]
    test_points, test_labels = points[split2:], labels[split2:]

    train_loader = prepare_dataloader(train_points, train_labels, args.batch_size)
    val_loader = prepare_dataloader(val_points, val_labels, args.batch_size)
    test_loader = prepare_dataloader(test_points, test_labels, args.batch_size)

    model = PointCloudNet(num_classes).to(device)

    start_time = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader, args.epochs, args.learning_rate, device)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.load_state_dict(torch.load('best_model.pth'))
    all_preds, all_labels = evaluate_model(model, test_loader, device)

    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    visualize_results(train_losses, val_losses, all_preds, all_labels, class_names)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model.pth')}")

    plt.savefig(os.path.join(args.output_dir, 'results.png'))
    print(f"Results visualization saved to {os.path.join(args.output_dir, 'results.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Classification")
    parser.add_argument('--data_path', type=str, default='data/raw_point_cloud.txt',
                        help='Path to the point cloud data file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    args = parser.parse_args()

    main(args)