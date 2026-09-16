# 基于BCG与PPG的多模态生理信号融合算法研究

1.设计 1D UNet​ 架构，以心冲击图（BCG）信号为输入实现心电图（ECG）重构，编码器逐级下采样提取深层特征，解码器通过跳跃连接融合多尺度信息，测试集 MSE 0.0291、RMSE 0.1705、MAE 0.0642，PSNR达27.57 dB，SSIM为0.5508，验证了 非接触式ECG重构​ 的可行性。

2.采用 SmoothL1Loss、ReduceLROnPlateau 学习率调度、早停及 ModelCheckpoint 等优化策略，重构 ECG 与真实 ECG 的趋势相关性达 SRCC 0.5962、KRCC 0.4448、PLCC 0.5898，为无袖带血压监测​提供技术基础。

3.构建基于 Attention BiLSTM​ 的血压估算模型，以光电容积脉搏波（PPG）与 ECG 为双输入，设计6层堆叠双向LSTM并嵌入 跨模态注意力机制，联合回归 ABP、SBP、DBP 三个目标值，实现多任务同步输出。

4.搭建完整数据流水线，对原始 BCG/PPG/ECG 信号进行滑动窗口分片（窗口长度4096/1024）、Z-Score标准化及70%/10%/20%数据集划分，保存预处理参数用于推理阶段逆标准化。

注：BCG数据集在右侧Releases里
