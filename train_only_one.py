import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# 定义LSTM模型
class VideoLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VideoLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.reshape(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# 自定义数据集类
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
                frame = cv2.resize(frame, (224, 224))
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

        frames = frames.permute(0, 3, 1, 2)  # 改变维度顺序为 [seq_len, c, h, w]
        return frames, label


# 数据准备函数
def prepare_data(video_folder, label_file, target_class=0):
    label_dict = {}
    with open(label_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            video_class, class_label = line.strip().split()
            label_dict[video_class] = int(class_label) - 1

    train_paths, test_paths, train_labels, test_labels = [], [], [], []

    for video_class in os.listdir(video_folder):
        class_path = os.path.join(video_folder, video_class)
        if os.path.isdir(class_path):
            if video_class not in label_dict:
                continue

            if label_dict[video_class] == target_class:
                for video_name in os.listdir(class_path):
                    if video_name.endswith('.avi'):
                        video_path = os.path.join(class_path, video_name)
                        label = label_dict[video_class]
                        if np.random.rand() < 0.8:
                            train_paths.append(video_path)
                            train_labels.append(label)
                        else:
                            test_paths.append(video_path)
                            test_labels.append(label)

    return train_paths, test_paths, train_labels, test_labels


# 训练模型函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=3, device='cpu'):
    model.to(device)

    train_losses, train_accuracies = [], []  # 用于存储训练损失和准确率

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return train_losses, train_accuracies


# 绘制图表函数
def plot_results(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    video_folder = 'D:\\ai\\UCF101\\UCF-101'
    label_file = 'D:\\ai\\UCF101\\UCF-101\\classInd.txt'

    train_paths, test_paths, train_labels, test_labels = prepare_data(video_folder, label_file, target_class=0)

    train_dataset = VideoDataset(train_paths, train_labels, fixed_frame_count=100)
    test_dataset = VideoDataset(test_paths, test_labels, fixed_frame_count=100)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    input_size = 224 * 224 * 3
    hidden_size = 512
    num_classes = len(set(train_labels))
    model = VideoLSTMModel(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并记录损失和准确率
    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=3, device='cpu')

    # 绘制图表
    plot_results(train_losses, train_accuracies)


if __name__ == '__main__':
    main()
