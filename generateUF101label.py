import os

def generate_class_ind(label_folder, output_file):
    # 获取所有类别的文件夹名
    class_names = sorted(os.listdir(label_folder))  # 排序确保类名的一致性
    with open(output_file, 'w') as f:
        for idx, class_name in enumerate(class_names):
            if os.path.isdir(os.path.join(label_folder, class_name)):
                f.write(f'{class_name} {idx}\n')
                print(f"Class: {class_name}, Label: {idx}")  # 打印类名和标签

# 假设你的类目录是 'UCF101/train'，输出的文件名为 'classInd.txt'

label_folder = 'D:\\ai\\UCF101\\UCF-101\\'  # 这里填入你的视频数据集目录
output_file = 'D:\\ai\\UCF101\\UCF-101\\classInd.txt'  # 这里填入你想保存的输出路径
generate_class_ind(label_folder, output_file)
