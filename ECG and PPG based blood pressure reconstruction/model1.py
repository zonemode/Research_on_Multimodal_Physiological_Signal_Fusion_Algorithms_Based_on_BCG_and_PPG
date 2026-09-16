import tensorflow as tf
import keras
from keras.api.layers import Input, LSTM, Bidirectional, Attention, Dense, Concatenate, Dropout,BatchNormalization
from keras.api.models import Model
from keras.api.callbacks import Callback,ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.api.utils import plot_model
from keras.api.models import load_model,save_model
import joblib

# 加载数据
abp_data = pd.read_csv("./Datasets/ABPn.csv")
ppg_data = pd.read_csv("./Datasets/PPGn.csv")
ecg_data = pd.read_csv("./Datasets/ECGn.csv")
sbp_data = pd.read_csv("./Datasets/SBPn.csv")
dbp_data = pd.read_csv("./Datasets/DBPn.csv")

# 将数据拆分为特征（PPG 和 ECG）和目标变量（ABP、SBP、DBP）
X_ppg = ppg_data.values
X_ecg = ecg_data.values
y_abp = abp_data.values
y_sbp = sbp_data.values
y_dbp = dbp_data.values

# 规范化/标准化输入特征
ppg_scaler = StandardScaler()
ecg_scaler = StandardScaler()
X_ppg = ppg_scaler.fit_transform(X_ppg)
X_ecg = ecg_scaler.fit_transform(X_ecg)

# 将数据拆分为训练集、验证集和测试集
X_ppg_train, X_ppg_temp, X_ecg_train, X_ecg_temp, y_abp_train, y_abp_temp, y_sbp_train, y_sbp_temp, y_dbp_train, y_dbp_temp = train_test_split(
    X_ppg, X_ecg, y_abp, y_sbp, y_dbp, test_size=0.2, random_state=42)
X_ppg_val, X_ppg_test, X_ecg_val, X_ecg_test, y_abp_val, y_abp_test, y_sbp_val, y_sbp_test, y_dbp_val, y_dbp_test = train_test_split(
    X_ppg_temp, X_ecg_temp, y_abp_temp, y_sbp_temp, y_dbp_temp, test_size=0.5, random_state=42)

# 定义自定义回调以在训练期间打印损失
class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} - Total Loss: {logs['loss']:.4f}, ABP Loss: {logs['abp_output_loss']:.4f}, SBP Loss: {logs['sbp_output_loss']:.4f}, DBP Loss: {logs['dbp_output_loss']:.4f}")

# 定义 PPG 和 ECG 数据的输入形状
ppg_input = Input(shape=(1024, 1), name="ppg_input")
ecg_input = Input(shape=(1024, 1), name="ecg_input")

# 初始化列表以保存 LSTM 和 Attention 层
layers = []

# 创建 6 对双向 LSTM 和 Attention 图层，并带有额外的密集图层
for _ in range(6):
    # 双向 LSTM 层
    lstm = Bidirectional(LSTM(32, return_sequences=True))
    ppg_lstm = lstm(ppg_input)
    ecg_lstm = lstm(ecg_input)

    # Attention 层
    attention = Attention()
    ppg_attention = attention([ppg_lstm, ecg_lstm])
    ecg_attention = attention([ecg_lstm, ppg_lstm])

    # 用于改进特征提取的额外密集层
    ppg_attention = Dense(64, activation="relu")(ppg_attention)
    ecg_attention = Dense(64, activation="relu")(ecg_attention)

    # 将图层追加到列表中
    layers.extend([ppg_lstm, ecg_lstm, ppg_attention, ecg_attention])

# 连接所有 LSTM 和 Attention 层的输出
combined = Concatenate(axis=-1)(layers)

# 添加更多密集层以进行进一步的特征提取
combined = Dense(128, activation="relu")(combined)
combined = Dropout(0.5)(combined)

# 创建用于 ABP 预测的输出图层
abp_output = Dense(1, name="abp_output")(combined)

# 创建用于 SBP 预测的输出图层
sbp_output = Dense(1, name="sbp_output")(combined)

# 创建用于 DBP 预测的输出图层
dbp_output = Dense(1, name="dbp_output")(combined)

# 创建模型
model = Model(inputs=[ppg_input, ecg_input], outputs=[abp_output, sbp_output, dbp_output])

# 为每个输出配置适当的损失函数和指标来编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"abp_output": "mean_squared_error", "sbp_output": "mean_squared_error", "dbp_output": "mean_squared_error"},
    metrics={"abp_output": "mae", "sbp_output": "mae", "dbp_output": "mae"}
)

# 自定义回调，用于在训练期间打印损失
loss_callback = LossCallback()

# 创建回调以在每个周期后保存模型权重
checkpoint_cb = ModelCheckpoint(
    filepath="best_model.keras",
    save_weights_only=False,
    monitor='val_loss',          # 监控验证损失
    mode='min',                  # 最小化损失
    save_best_only=True,         # 只保存最佳模型
    verbose=1
)

# 使用自定义回调训练模型并打印损失
history = model.fit(
    [X_ppg_train, X_ecg_train],
    [y_abp_train, y_sbp_train, y_dbp_train],
    validation_data=([X_ppg_val, X_ecg_val], [y_abp_val, y_sbp_val, y_dbp_val]),
    epochs=10,
    batch_size=32,
    callbacks=[loss_callback, checkpoint_cb]  # 添加用于打印损失的定制回调函数
)
# 根据测试数据评估模型
loss, abp_loss, sbp_loss, dbp_loss, abp_mae, sbp_mae, dbp_mae = model.evaluate([X_ppg_test, X_ecg_test], [y_abp_test, y_sbp_test, y_dbp_test], verbose=2)
print(f"Total Loss: {loss}")
print(f"ABP Loss: {abp_loss}")
print(f"SBP Loss: {sbp_loss}")
print(f"DBP Loss: {dbp_loss}")
print(f"ABP MAE: {abp_mae}")
print(f"SBP MAE: {sbp_mae}")
print(f"DBP MAE: {dbp_mae}")

abp_pred, sbp_pred, dbp_pred = model.predict([X_ppg_test, X_ecg_test])

abp_std = np.std(abp_pred)
sbp_std = np.std(sbp_pred)
dbp_std = np.std(dbp_pred)

print(f"ABP std: {abp_std}")
print(f"SBP std: {sbp_std}")
print(f"DBP std: {dbp_std}")

# 打印模型结构摘要
model.summary()

# 可视化模型结构
plot_model(model, to_file='model_architecture.png', show_shapes=True)

# 可视化训练历史记录
plt.figure(figsize=(12, 6))
#plt.plot(history.history['abp_output_loss'], label='ABP Loss')
plt.plot(history.history['sbp_output_loss'], label='SBP Loss')
plt.plot(history.history['dbp_output_loss'], label='DBP Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("blood_pressure_model.keras")
joblib.dump(ppg_scaler, 'ppg_scaler.pkl')
joblib.dump(ecg_scaler, 'ecg_scaler.pkl')