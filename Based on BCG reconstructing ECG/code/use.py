# 导入必要的库
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import matplotlib.pyplot as plt  # 绘图

# 设置计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义1D UNet模型结构（必须与训练时相同）
class Conv1DUNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 编码器部分
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)  # 瓶颈层
        
        # 解码器部分
        self.up4 = self.upconv_block(1024, 512)
        self.up3 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.final_conv = torch.nn.Conv1d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.nn.functional.max_pool1d(enc1, 2))
        enc3 = self.enc3(torch.nn.functional.max_pool1d(enc2, 2))
        enc4 = self.enc4(torch.nn.functional.max_pool1d(enc3, 2))
        bottleneck = self.bottleneck(torch.nn.functional.max_pool1d(enc4, 2))
        
        up4 = self.up4(bottleneck)
        up4 = up4 + enc4
        up3 = self.up3(up4)
        up3 = up3 + enc3
        up2 = self.up2(up3)
        up2 = up2 + enc2
        up1 = self.up1(up2)
        up1 = up1 + enc1
        
        output = self.final_conv(up1)
        return output

# 加载模型
def load_unet_model(model_path):
    """加载预训练的UNet模型"""
    model = Conv1DUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model

# 数据预处理函数
def batch_list(data, instant_size=4096):
    batches = [data[i:i + instant_size] for i in range(0, len(data), instant_size)]
    if len(batches[-1]) < instant_size:
        batches.pop()
    return batches

# 标准化函数
def standardize(data, mean, std):
    return (data - mean) / std

# 反标准化函数
def inverse_standardize(data, mean, std):
    return data * std + mean


# 主函数
def main():
    # 1. 加载预训练模型
    model_path = "result/1_unet/unet_model.pth"  # 模型路径
    unet_model = load_unet_model(model_path)
    print("模型加载完成")
    
    # 2. 准备新数据（从CSV文件加载）
    csv_path = ".pats/pat1.csv"  # 你的数据文件路径
    csv_files = glob.glob(csv_path)
    
    # 加载标准化参数（使用训练时的均值/标准差）
    params = np.load("result/1_unet/normalization_params.npy", allow_pickle=True).item()
    x_mean, x_std = params['x_mean'], params['x_std']
    
    # 3. 处理并预测新数据
    all_predictions = []
    
    for file in csv_files:
        # 读取CSV数据
        data = pd.read_csv(file)
        bcg_data = data['LC_BCG3'].values
        
        # 标准化数据（使用训练时的统计量）
        bcg_data_std = standardize(bcg_data, x_mean, x_std)
        
        # 分批处理数据
        batches = batch_list(bcg_data_std)
        batch_tensor = torch.tensor(batches, dtype=torch.float32)[:, np.newaxis, :]  # 添加通道维度
        
        # 创建DataLoader
        dataset = TensorDataset(batch_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)  # 批量大小根据GPU内存调整
        
        # 预测批次
        file_predictions = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                outputs = unet_model(batch)
                file_predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测
        file_predictions = np.concatenate(file_predictions, axis=0)
        all_predictions.append(file_predictions)
        print(f"完成文件预测: {file}")
    
    # 4. 后处理结果（反标准化）
    y_mean, y_std = params['y_mean'], params['y_std']
    
    # 将预测结果反标准化为原始单位
    final_predictions = inverse_standardize(np.concatenate(all_predictions), y_mean, y_std)

    #plot_comparison(torch.tensor(final_predictions), 0,"predictions")
    
    # 5. 保存预测结果
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取ECG预测值
    ecg_predictions_list = []
    
    # 遍历所有预测样本
    for sample_predictions in final_predictions[:1]:
        # 每个sample_predictions的形状为 (1, 4096)
        ecg_signal = sample_predictions[0].tolist()  # 转换为Python列表
        ecg_predictions_list.append(ecg_signal)
    
    # 保存为JSON文件
    json_path = os.path.join(output_dir, "ecg_predictions2.json")
    with open(json_path, 'w') as json_file:
        # 只保存ECG信号列表
        json.dump(ecg_predictions_list, json_file, indent=2)
    
    print(f"预测完成! ECG结果已保存至: {json_path}")
    
    # 可选：同时保存为numpy格式
    np.save(os.path.join(output_dir, "ecg_predictions2.npy"), final_predictions)
    print(f"Numpy格式备份已保存")
    
    # 保存为CSV文件（如果需要）
    # 将数据合并回原始格式（需要相应处理）
    
    print(f"预测完成! 结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()