import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import torch

# 定义LSTM + Transformer模型
import torchvision.models as models

class VideoLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VideoLSTMTransformerModel, self).__init__()
        # 使用预训练的ResNet18模型作为特征提取器
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉最后的全连接层，保留特征提取部分

        # 定义LSTM层
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)  # ResNet18输出512维的特征

        # 定义Transformer层
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=4, num_decoder_layers=4,
                                          batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            frame_features = self.resnet(frame)
            features.append(frame_features.unsqueeze(1))

        features = torch.cat(features, dim=1)

        lstm_out, _ = self.lstm(features)
        transformer_out = self.transformer(lstm_out, lstm_out)
        out = self.fc(transformer_out[:, -1, :])

        return out


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, fixed_frame_count=100, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.fixed_frame_count = fixed_frame_count
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))  # 统一调整帧大小
                frames.append(frame)
            else:
                break
        cap.release()

        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0

        if len(frames) < self.fixed_frame_count:
            padding = self.fixed_frame_count - len(frames)
            frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif len(frames) > self.fixed_frame_count:
            frames = frames[:self.fixed_frame_count]

        frames = torch.tensor(frames, dtype=torch.float32).clone().detach()
        frames = frames.permute(0, 3, 1, 2)

        return frames, label


class VideoBehaviorModel:
    def __init__(self, video_folder, input_size=224, hidden_size=256, num_classes=1, batch_size=8,
                 epochs=3, fixed_frame_count=100, device='cpu'):
        self.video_folder = video_folder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.fixed_frame_count = fixed_frame_count
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.model = None

    def prepare_data(self):
        """
        只加载 'HighJump' 类别的视频数据
        """
        video_paths, labels = [], []

        # 只选择 'HighJump' 类别
        highjump_class_path = os.path.join(self.video_folder, 'HighJump')
        if not os.path.exists(highjump_class_path):
            raise ValueError("No 'HighJump' class found in the dataset")

        # 获取HighJump类的视频路径和标签
        for video_name in os.listdir(highjump_class_path):
            if video_name.endswith('.avi'):  # 只处理.avi格式的视频
                video_path = os.path.join(highjump_class_path, video_name)
                video_paths.append(video_path)
                labels.append(0)  # 'HighJump' 类别设为标签0

        # 将数据划分为训练集和测试集
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            video_paths, labels, test_size=0.2, random_state=42, stratify=labels)

        print(f"Training samples: {len(train_paths)}")
        print(f"Testing samples: {len(test_paths)}")

        return train_paths, test_paths, train_labels, test_labels

    def create_data_loaders(self, train_paths, test_paths, train_labels, test_labels):
        train_dataset = VideoDataset(train_paths, train_labels, fixed_frame_count=self.fixed_frame_count)
        test_dataset = VideoDataset(test_paths, test_labels, fixed_frame_count=self.fixed_frame_count)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader

    def train_model(self, model, train_loader, criterion, optimizer):
        model.to(self.device)
        train_losses, train_accuracies = [], []
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            train_losses.append(running_loss / len(train_loader))
            train_accuracies.append(accuracy)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        return train_losses, train_accuracies

    def test_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, model, path='video_behavior_model.pth'):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def plot_results(self, train_losses, train_accuracies):
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.show()

    def main(self):
        # 准备数据
        train_paths, test_paths, train_labels, test_labels = self.prepare_data()

        # 创建数据加载器
        train_loader, test_loader = self.create_data_loaders(train_paths, test_paths, train_labels, test_labels)

        # 初始化模型
        model = VideoLSTMTransformerModel(input_size=224, hidden_size=256, num_classes=1)  # 类别只有HighJump
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        train_losses, train_accuracies = self.train_model(model, train_loader, criterion, optimizer)

        # 测试模型
        test_accuracy = self.test_model(model, test_loader)

        # 保存模型
        self.save_model(model, 'highjump_video_behavior_model.pth')

        # 绘制训练结果
        self.plot_results(train_losses, train_accuracies)


if __name__ == '__main__':
    video_folder = 'D:\\ai\\行为预测\\UCF-101'  # 你的数据集路径
    model = VideoBehaviorModel(video_folder, device='cpu')
    model.main()






