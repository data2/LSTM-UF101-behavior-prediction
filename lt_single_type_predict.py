import torch
import cv2
import numpy as np
from torchvision import transforms
from lt_single_type import VideoLSTMTransformerModel  # 如果有模块化，导入模型类

# 加载训练好的模型
def load_model(model_path, input_size=224, hidden_size=256, num_classes=1, device='cpu'):
    model = VideoLSTMTransformerModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# 处理输入视频，确保视频帧数量统一并进行必要的预处理
def process_video(video_path, fixed_frame_count=100):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # 读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))  # 统一调整帧大小
            frames.append(frame)
        else:
            break
    cap.release()

    frames = np.array(frames)

    # 打印出原始frames的维度
    print("Original frames shape:", frames.shape)

    frames = torch.tensor(frames, dtype=torch.float32) / 255.0  # 归一化

    # 处理帧数不足fixed_frame_count的情况
    if frames.shape[0] < fixed_frame_count:
        padding = fixed_frame_count - frames.shape[0]
        frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        print(f"Padded frames shape (less frames): {frames.shape}")
    elif frames.shape[0] > fixed_frame_count:
        frames = frames[:fixed_frame_count]
        print(f"Trimmed frames shape (more frames): {frames.shape}")

    # 确保 frames 的维度顺序为 [Seq_len, C, H, W]
    frames = torch.tensor(frames, dtype=torch.float32).clone().detach()
    frames = frames.permute(0, 3, 1, 2)  # [num_frames, H, W, C] -> [num_frames, C, H, W]

    # 打印最终的frames形状
    print("Final frames shape after padding/trim:", frames.shape)

    return frames.unsqueeze(0)  # 添加批次维度 [1, Seq_len, C, H, W]


# 进行视频预测，判断视频是否为“跳高”
def predict(model, video_path, device='cpu'):
    # 处理视频
    frames = process_video(video_path)

    # 将数据转到正确的设备
    frames = frames.to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(frames)
        predicted = torch.sigmoid(outputs)  # 对输出进行sigmoid激活函数处理，获取概率值
        prediction = predicted.item()

    # 如果预测值大于某个阈值（如 0.5），认为是跳高
    if prediction > 0.5:
        print(f"The video '{video_path}' contains 'HighJump' with confidence {prediction:.2f}.")
    else:
        print(f"The video '{video_path}' does not contain 'HighJump' with confidence {prediction:.2f}.")

# 主函数
def main():
    model_path = 'highjump_video_behavior_model.pth'  # 你的训练模型路径
    # video_path = 'D:\\ai\\行为预测\\UCF-101\\HighJump\\v_HighJump_g17_c03.avi'  # 你要测试的视频路径
    video_path = 'D:\\ai\\行为预测\\UCF-101\\HandstandWalking\\v_HandstandWalking_g04_c01.avi'  # 你要测试的视频路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练好的模型
    model = load_model(model_path, device=device)

    # 对视频进行预测
    predict(model, video_path, device=device)

if __name__ == '__main__':
    main()
