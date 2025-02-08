import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 定义LSTM + Transformer模型
class VideoLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        初始化LSTM和Transformer模型
        input_size: 每帧的输入特征维度
        hidden_size: LSTM的隐藏层维度
        num_classes: 任务的分类数（UCF-101有101个动作类别）
        """
        super(VideoLSTMTransformerModel, self).__init__()

        # 定义LSTM层，输入维度为每帧的特征，隐藏层维度为hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 定义Transformer层，处理序列数据
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                          batch_first=True)

        # 输出层，将Transformer的输出转换为分类结果
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播
        x: 输入数据，形状为 [batch_size, seq_len, c, h, w]
        """
        batch_size, seq_len, c, h, w = x.size()

        # 将每帧图像展平为一维，变为 [batch_size, seq_len, c*h*w]
        x = x.reshape(batch_size, seq_len, -1)

        # LSTM层的输出
        lstm_out, _ = self.lstm(x)

        # 使用Transformer进行序列建模
        transformer_out = self.transformer(lstm_out, lstm_out)

        # 取Transformer的最后一个输出用于分类
        out = self.fc(transformer_out[:, -1, :])

        return out


# 自定义数据集类
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, fixed_frame_count=100, transform=None):
        """
        初始化视频数据集
        video_paths: 视频路径列表
        labels: 标签列表
        fixed_frame_count: 固定的帧数，统一长度的帧序列
        transform: 可选的数据变换
        """
        self.video_paths = video_paths
        self.labels = labels
        self.fixed_frame_count = fixed_frame_count
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        idx: 样本的索引
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 读取视频
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

        # 转换为numpy数组并归一化到[0, 1]区间
        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0

        # 如果帧数不足fixed_frame_count，进行零填充
        if len(frames) < self.fixed_frame_count:
            padding = self.fixed_frame_count - len(frames)
            frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif len(frames) > self.fixed_frame_count:
            frames = frames[:self.fixed_frame_count]  # 截取固定长度的帧

        # 确保tensor不共享计算图并改变维度顺序 [seq_len, c, h, w]
        frames = torch.tensor(frames, dtype=torch.float32).clone().detach()
        frames = frames.permute(0, 3, 1, 2)  # 变换维度顺序为 [seq_len, c, h, w]

        return frames, label


# 视频行为预测模型
class VideoBehaviorModel:
    def __init__(self, video_folder, input_size=224, hidden_size=512, num_classes=101, batch_size=8,
                 epochs=3, fixed_frame_count=100, device='cpu'):
        """
        初始化视频行为预测模型
        video_folder: 视频数据集文件夹路径
        input_size: 输入帧的大小（224x224）
        hidden_size: LSTM隐藏层大小
        num_classes: 分类数目
        batch_size: 批处理大小
        epochs: 训练周期
        fixed_frame_count: 固定帧数（视频长度）
        device: 使用的设备，如'cpu'或'cuda'
        """
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
        准备数据，获取视频路径和标签
        将数据集拆分为训练集和测试集
        """
        video_paths, labels = [], []

        # 获取所有类别的子文件夹名
        class_names = [folder for folder in os.listdir(self.video_folder) if
                       os.path.isdir(os.path.join(self.video_folder, folder))]
        class_names.sort()

        # 为每个类别分配一个标签
        label_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

        # 获取每个视频路径及其对应标签
        for video_class, label in label_dict.items():
            class_path = os.path.join(self.video_folder, video_class)
            for video_name in os.listdir(class_path):
                if video_name.endswith('.avi'):  # 只处理.avi格式的视频
                    video_path = os.path.join(class_path, video_name)
                    video_paths.append(video_path)
                    labels.append(label)

        # 将数据划分为训练集和测试集
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            video_paths, labels, test_size=0.2, random_state=42, stratify=labels)

        print(f"Training samples: {len(train_paths)}")
        print(f"Testing samples: {len(test_paths)}")

        return train_paths, test_paths, train_labels, test_labels

    def create_data_loaders(self, train_paths, test_paths, train_labels, test_labels):
        """
        创建训练集和测试集的DataLoader
        """
        train_dataset = VideoDataset(train_paths, train_labels, fixed_frame_count=self.fixed_frame_count)
        test_dataset = VideoDataset(test_paths, test_labels, fixed_frame_count=self.fixed_frame_count)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader

    def train_model(self, model, train_loader, criterion, optimizer):
        """
        训练模型
        """
        model.to(self.device)
        train_losses, train_accuracies = [], []

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        return train_losses, train_accuracies

    def test_model(self, model, test_loader):
        """
        测试模型
        """
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
        """
        保存训练好的模型
        """
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def plot_results(self, train_losses, train_accuracies):
        """
        绘制训练损失和准确率图表
        """
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
        """
        主训练流程
        """
        train_paths, test_paths, train_labels, test_labels = self.prepare_data()
        train_loader, test_loader = self.create_data_loaders(train_paths, test_paths, train_labels, test_labels)

        # 创建LSTM + Transformer模型
        model = VideoLSTMTransformerModel(input_size=224 * 224 * 3, hidden_size=self.hidden_size,
                                          num_classes=self.num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        train_losses, train_accuracies = self.train_model(model, train_loader, criterion, optimizer)

        # 测试模型
        self.test_model(model, test_loader)

        # 保存模型
        self.save_model(model)

        # 绘制训练过程的图表
        self.plot_results(train_losses, train_accuracies)


if __name__ == '__main__':
    # 修改为你的数据集路径
    video_folder = 'D:\\ai\\行为预测\\UCF-101'
    model = VideoBehaviorModel(video_folder, device='cpu')
    model.main()

