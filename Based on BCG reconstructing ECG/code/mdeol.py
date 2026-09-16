import glob  # 文件路径匹配
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import torch.nn.functional as F  # PyTorch神经网络函数
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import DataLoader, TensorDataset  # 数据加载工具
import torch.optim as optim  # 优化器
import matplotlib.pyplot as plt  # 绘图
import time  # 时间记录
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # 图像质量评估指标
from scipy.stats import spearmanr, kendalltau, pearsonr  # 统计相关性指标
import os  # 操作系统接口

start_time = time.time() # 记录开始时间

# 超参数设置
instant_size = 4096  # 每个数据样本的长度
batch_size = 128     # 批量大小
epochs = 200         # 训练轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 计算设备选择
csv_path = ".pats/*pred.csv"  # 数据文件路径模式
csv_files = glob.glob(csv_path)  # 获取匹配的文件列表
save_dir =  "result/1_unet"  # 结果保存路径

# 数据读取与预处理
all_ecg_data = []   # 存储所有ECG数据
all_bcg_data = []   # 存储所有BCG数据

# 遍历CSV文件并加载数据
for i, file in enumerate(csv_files):
    data = pd.read_csv(file)
    all_ecg_data.append(data['ECG'].values)       # 提取ECG列数据
    all_bcg_data.append(data['LC_BCG3'].values)   # 提取BCG列数据
    
print("数据读取完成, 用时:", time.time() - start_time)

# 数据集划分（按70%-10%-20%比例）
x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
for i, ecg_sample in enumerate(all_ecg_data):
    total_len = len(ecg_sample)
    train_idx = int(0.7 * total_len)  # 训练集截止索引
    val_idx = int(0.8 * total_len)    # 验证集截止索引
    bcg_sample = all_bcg_data[i]      # 对应BCG数据
    
    # 划分数据集
    x_train.extend(bcg_sample[:train_idx])
    y_train.extend(ecg_sample[:train_idx])
    x_val.extend(bcg_sample[train_idx:val_idx])
    y_val.extend(ecg_sample[train_idx:val_idx])
    x_test.extend(bcg_sample[val_idx:])
    y_test.extend(ecg_sample[val_idx:])

# 数据标准化函数
def standardize(data, mean, std):
    """标准化数据"""
    return (data - mean) / std

def inverse_standardize(data, mean, std):
    """反标准化数据"""
    return data * std + mean

# 计算标准化参数
x_train_np, y_train_np = np.array(x_train), np.array(y_train)
x_val_np, y_val_np = np.array(x_val), np.array(y_val)
x_test_np, y_test_np = np.array(x_test), np.array(y_test)
x_mean, x_std = x_train_np.mean(), x_train_np.std()
y_mean, y_std = y_train_np.mean(), y_train_np.std()

# 应用标准化
x_train = standardize(x_train_np, x_mean, x_std)
y_train = standardize(y_train_np, y_mean, y_std)
x_val = standardize(x_val_np, x_mean, x_std)
y_val = standardize(y_val_np, y_mean, y_std)
x_test = standardize(x_test_np, x_mean, x_std)
y_test = standardize(y_test_np, y_mean, y_std)

def batch_list(data, instant_size):
    """将长序列分割为固定长度的片段"""
    batches = [data[i:i + instant_size] for i in range(0, len(data), instant_size)]
    if len(batches[-1]) < instant_size:  # 丢弃最后不足一个批次的片段
        batches.pop()
    return batches

# 数据批处理与维度调整（添加通道维度）
x_train = np.array(batch_list(x_train, instant_size))[:, np.newaxis, :]  # 形状变为(N, 1, 4096)
y_train = np.array(batch_list(y_train, instant_size))[:, np.newaxis, :]
x_val = np.array(batch_list(x_val, instant_size))[:, np.newaxis, :]
y_val = np.array(batch_list(y_val, instant_size))[:, np.newaxis, :]
x_test = np.array(batch_list(x_test, instant_size))[:, np.newaxis, :]
y_test = np.array(batch_list(y_test, instant_size))[:, np.newaxis, :]

# 转换为PyTorch张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义1D UNet模型
class Conv1DUNet(nn.Module):
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
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        """卷积块：两个卷积层 + 批归一化 + ReLU"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        """上采样块：转置卷积 + 卷积层"""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码过程
        enc1 = self.enc1(x)  # (B, 64, 4096)
        enc2 = self.enc2(F.max_pool1d(enc1, 2))  # (B, 128, 2048)
        enc3 = self.enc3(F.max_pool1d(enc2, 2))  # (B, 256, 1024)
        enc4 = self.enc4(F.max_pool1d(enc3, 2))  # (B, 512, 512)
        bottleneck = self.bottleneck(F.max_pool1d(enc4, 2))  # (B, 1024, 256)
        
        # 解码过程（包含跳跃连接）
        up4 = self.up4(bottleneck)  # (B, 512, 512)
        up4 = up4 + enc4  # 跳跃连接
        up3 = self.up3(up4)  # (B, 256, 1024)
        up3 = up3 + enc3
        up2 = self.up2(up3)  # (B, 128, 2048)
        up2 = up2 + enc2
        up1 = self.up1(up2)  # (B, 64, 4096)
        up1 = up1 + enc1
        
        output = self.final_conv(up1)  # (B, 1, 4096)
        return output

# 初始化模型、损失函数和优化器
model = Conv1DUNet(in_channels=1, out_channels=1).to(device)
criterion = nn.SmoothL1Loss(beta=1.0)  # 使用平滑L1损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  # 动态调整学习率

# 训练循环
model.train()
train_losses = []  # 记录训练损失
val_losses = []    # 记录验证损失
best_val_loss = float('inf')
patience = 10       # 早停等待周期
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:  # 遍历训练集
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停与模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/unet_model.pth")  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(val_loss)  # 调整学习率

# 加载最佳模型进行测试
model.load_state_dict(torch.load(f"{save_dir}/unet_model.pth"))
model.eval()

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'loss_curves_two_level.png'))
plt.close()

# 定义结果可视化函数
def plot_comparison(ecg_true, ecg_pred, index, save_dir):
    """绘制真实与预测ECG对比图"""
    plt.figure(figsize=(12, 6))
    for i in range(min(4, ecg_true.shape[0])):
        plt.subplot(2, 2, i+1)
        plt.plot(ecg_true[i, 0].cpu().numpy(), label='Ground Truth')
        plt.plot(ecg_pred[i, 0].cpu().numpy(), label='Reconstructed', linestyle='--')
        plt.legend()
        plt.title(f'Sample {i}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'compare_batch_{index}.png'))
    plt.close()

# 定义评估指标计算函数
def evaluate_metrics(y_true, y_pred):
    """计算并返回多个评估指标"""
    metrics = {}
    
    y_true = y_true[:, 0, :]  # 去除通道维度
    y_pred = y_pred[:, 0, :]
    
    # 展平数据用于相关性计算
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # MSE/RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    metrics['MSE'] = mse
    metrics['RMSE'] = np.sqrt(mse)
    metrics['rRMSE'] = metrics['RMSE'] / (y_true.max() - y_true.min())
    
    # MAE
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    
    # PSNR/SSIM
    metrics['PSNR'] = peak_signal_noise_ratio(y_true, y_pred, data_range=y_true.max() - y_true.min())
    ssim_values = [structural_similarity(y_true[i], y_pred[i], data_range=y_true[i].max() - y_true[i].min()) 
                   for i in range(y_true.shape[0])]
    metrics['SSIM'] = np.mean(ssim_values)
    
    # 相关性指标
    metrics['SRCC'], _ = spearmanr(y_true_flat, y_pred_flat)
    metrics['KRCC'], _ = kendalltau(y_true_flat, y_pred_flat)
    metrics['PLCC'], _ = pearsonr(y_true_flat, y_pred_flat)
    
    return metrics

np.save("result/1_unet/normalization_params.npy", {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std})

# 测试模型
model.eval()
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        
        # 反标准化
        batch_y_raw = inverse_standardize(batch_y.cpu().numpy(), y_mean, y_std)
        outputs_raw = inverse_standardize(outputs.cpu().numpy(), y_mean, y_std)
        
        y_true_all.append(batch_y_raw)
        y_pred_all.append(outputs_raw)
        
        plot_comparison(torch.tensor(batch_y_raw), torch.tensor(outputs_raw), batch_idx, save_dir)

# 合并所有测试结果
y_true_all = np.concatenate(y_true_all, axis=0)
y_pred_all = np.concatenate(y_pred_all, axis=0)

# 计算并保存指标
metrics = evaluate_metrics(y_true_all, y_pred_all)
print("Test Metrics (UNet Version):")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

with open(os.path.join(save_dir, 'test_metrics_unet.txt'), 'w') as f:
    for key in metrics:
        f.write(f"{key}: {metrics[key]:.4f}\n")

print("Total time:", time.time() - start_time)