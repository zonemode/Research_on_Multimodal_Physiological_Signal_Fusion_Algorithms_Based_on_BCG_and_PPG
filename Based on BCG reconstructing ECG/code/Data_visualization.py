import glob
import pandas as pd
import matplotlib.pyplot as plt
# 设置CSV文件的路径
# CSV_BCG_1 = "D:\ZYT\lab\Dataset Files\Kensas\.pats\pat1.csv"
CSV_BCG_1_PRED = "D:\ZYT\lab\Dataset Files\Kensas\.pats\pat2pred.csv"

# 读取数据
# data_bcg_1 = pd.read_csv(CSV_BCG_1)
data_bcg_1_pred = pd.read_csv(CSV_BCG_1_PRED)

# 提取BCG数据
# bcg_1 = data_bcg_1['LC_BCG0'].values
bcg_1_pred = data_bcg_1_pred['LC_BCG0'].values
# rebap = data_bcg_1_pred['reBAP'].values
ecg = data_bcg_1_pred['ECG'].values

# # 可视化BCG数据,分为两幅图,设置x坐标为1000到5000
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.xlim(0, 5000)
# plt.plot(bcg_1)
# plt.title("BCG Data")
# plt.subplot(2, 1, 2)
# plt.plot(bcg_1_pred)
# plt.xlim(0, 5000)
# plt.title("BCG Data preprocessed")
# plt.tight_layout()
# plt.show()

# # 可视化rebap的数据，一幅图，x坐标标题为"Time (ms)"，y坐标标题为"ReBAP Amplitude"，x坐标范围为1000到5000
# plt.figure(figsize=(12, 6))
# # 颜色设置为红色
# plt.plot(rebap, color='red')
# plt.xlim(2000, 4000)
# # plt.ylim(0.6, 1.3)
# plt.xlabel("Time (ms)")
# plt.ylabel("ReBAP Amplitude")
# plt.title("ReBAP Data")
# # plt.show()
# plt.savefig(r"D:\ZYT\WJS\2025.4.5项目申请\血压图片\rebap_5_pred.png", dpi=300, bbox_inches='tight')




# 可视化bcg_1_pred的数据，一幅图，x坐标标题为"Time (ms)"，y坐标标题为"BCG Amplitude"，x坐标范围为1000到5000
plt.figure(figsize=(12, 6))
plt.plot(ecg, color = 'red')
plt.xlim(2000, 3000)
# plt.ylim(-0.02,0.02)
plt.xlabel("Time (ms)")
plt.ylabel("ECG Amplitude")
plt.title("ECG Data preprocessed")
# plt.show()
plt.savefig(r"D:\ZYT\lab\科研配图\素材\ECG图片\bcg_5_pred.png", dpi=300, bbox_inches='tight')