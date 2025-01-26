import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


# 定义LSTM模型
class VideoLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VideoLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    # 在 forward 方法中修改
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # [batch_size, seq_len, c, h, w]
        x = x.reshape(batch_size, seq_len, -1)  # 使用 reshape 代替 view
        lstm_out, _ = self.lstm(x)  # LSTM输出
        out = self.fc(lstm_out[:, -1, :])  # 取LSTM最后一个时刻的输出
        return out


# 自定义数据集类
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, fixed_frame_count=100, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.fixed_frame_count = fixed_frame_count  # 固定帧数
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 读取视频文件
        cap = cv2.VideoCapture(video_path)
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))  # 调整帧大小
                frames.append(frame)
            else:
                break
        cap.release()

        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0  # 归一化

        # 如果视频帧数小于固定帧数，填充；如果大于，裁剪
        if len(frames) < self.fixed_frame_count:
            padding = self.fixed_frame_count - len(frames)
            frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)  # 填充
        elif len(frames) > self.fixed_frame_count:
            frames = frames[:self.fixed_frame_count]  # 裁剪

        frames = torch.tensor(frames, dtype=torch.float32)  # 归一化
        frames = frames.permute(0, 3, 1, 2)  # 改变维度顺序为 [seq_len, c, h, w]

        return frames, label


# 数据准备函数
def prepare_data(video_folder, label_file):
    # 创建一个字典，将类名映射到整数标签
    label_dict = {}

    with open(label_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            video_class, class_label = line.strip().split()
            label_dict[video_class] = int(class_label) -1  # 使用标签作为整数

    # 打印 label_dict 检查是否正确填充
    print("label_dict:", label_dict)

    # 准备训练和测试数据的列表
    train_paths, test_paths, train_labels, test_labels = [], [], [], []

    # 遍历视频文件夹，加载视频路径及其对应的标签
    for video_class in os.listdir(video_folder):
        class_path = os.path.join(video_folder, video_class)
        if os.path.isdir(class_path):  # 检查是否是文件夹
            print(f"Processing class: {video_class}")  # 打印正在处理的类
            if video_class not in label_dict:
                print(f"Warning: Class '{video_class}' not found in label_dict")
                continue  # 如果类名在字典中没有找到对应标签，跳过该类

            for video_name in os.listdir(class_path):
                if video_name.endswith('.avi'):
                    video_path = os.path.join(class_path, video_name)
                    label = label_dict[video_class]  # 获取整数标签
                    # 按照一定比例（例如80-20）划分数据集为训练集和测试集
                    if np.random.rand() < 0.8:  # 训练集
                        train_paths.append(video_path)
                        train_labels.append(label)
                    else:  # 测试集
                        test_paths.append(video_path)
                        test_labels.append(label)

    print(f"Total train videos: {len(train_paths)}")
    print(f"Total test videos: {len(test_paths)}")

    return train_paths, test_paths, train_labels, test_labels


# 训练模型函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
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

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total * 100:.2f}%')


# 主函数
def main():
    video_folder = 'D:\\ai\\UCF101\\UCF-101'  # 替换为视频文件夹路径
    label_file = 'D:\\ai\\UCF101\\UCF-101\\classInd.txt'  # 替换为标签文件路径

    # 准备数据
    train_paths, test_paths, train_labels, test_labels = prepare_data(video_folder, label_file)

    # 创建数据集和数据加载器
    train_dataset = VideoDataset(train_paths, train_labels, fixed_frame_count=100)  # 固定帧数设置为100
    test_dataset = VideoDataset(test_paths, test_labels, fixed_frame_count=100)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 定义模型、损失函数和优化器
    input_size = 224 * 224 * 3  # 每帧的大小
    hidden_size = 512  # LSTM的隐藏层大小
    num_classes = len(set(train_labels))  # 标签的数量
    model = VideoLSTMModel(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu')


if __name__ == '__main__':
    main()
