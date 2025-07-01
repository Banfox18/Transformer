import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
import matplotlib.pyplot as plt

def calculate_bollinger_bands(data, window=10, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_rsi(data, window=10):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_roc(close, periods=14):  
    return close.pct_change(periods=periods, fill_method=None) * 100  

stats = {}
ticker = 'daily' 

data = pd.read_csv('Data/CSI 300/' + ticker + '.csv')
# 提取参数
openPrice = data['OpenPrice']
highPrice = data['HighPrice']
lowPrice = data['LowPrice']
closePrice = data['ClosePrice']
volume = data['Volume']
changeRatio = data['ChangeRatio']
data['Date'] = pd.to_datetime(data['CloseDate'])

# 计算技术指标
upperBollBands, lowerBollBands = calculate_bollinger_bands(closePrice, window=14, num_of_std=2)
bollWidth = upperBollBands - lowerBollBands
rsi = calculate_rsi(closePrice, window=14)
roc = calculate_roc(closePrice, periods=10)

dataFrame = pd.DataFrame({
    # 'OpenPrice': openPrice,
    # 'HighPrice': highPrice,
    # 'LowPrice': lowPrice,
    'ClosePrice': closePrice,
    'Volume': volume,
    'ChangeRatio': changeRatio,
    'BollWidth': bollWidth,
    'RSI': rsi,
    'ROC': roc,
    'Date': data['Date']
})
# 设置日期为索引
dataFrame.set_index('Date', inplace=True)
dataFrame.head()

# 标准化数据, 剔除所有数据包含NaN的行
dataFrame.dropna(inplace=True)
MEAN = dataFrame.mean()
STD = dataFrame.std()

for column in MEAN.index:
    stats[f'{column}_mean'] = MEAN[column]
    stats[f'{column}_std'] = STD[column]

dataFrame = (dataFrame - MEAN) / STD
dataFrame.dropna(inplace=True)
dataFrame.head(500)

stats = pd.DataFrame([stats], index = [0])
stats.head()

# 切片原始数据并打乱
labels = dataFrame.shift(-1)
dataFrame = dataFrame.iloc[:-1]
labels = labels.iloc[:-1]

SEQUENCE_LEN = 20
PREDICTION_LEN = 10

def createSequences(data, labels, mean, std, sequenceLength=SEQUENCE_LEN, predictionLenth=PREDICTION_LEN):
    sequences = []
    label = []
    dataSize = len(data)

    for i in range(dataSize - sequenceLength - predictionLenth - 1):
        if i == 0:
            continue
        sequences.append(data[i:i + sequenceLength])
        label.append([labels[i - 1], labels[i + predictionLenth], mean[0], std[0]])

    return np.array(sequences), np.array(label) # 这两个都是三维数组

data = np.column_stack((dataFrame['ClosePrice'].values,
                        dataFrame['BollWidth'].values,
                        dataFrame['RSI'].values,
                        dataFrame['ROC'].values,
                        dataFrame['Volume'].values,
                        dataFrame['ChangeRatio'].values))
sequences, label = createSequences(data, 
                                   labels['ClosePrice'].values[SEQUENCE_LEN - 1:], # 切片操作, 表示从第 SEQUENCE_LEN - 1 个元素开始提取直到末尾, 为了对齐sequence和label的结束位置
                                   stats['ClosePrice_mean'].values,
                                   stats['ClosePrice_std'].values)

print(data)

np.random.seed(112300)
# 数据清洗
shuffledIndices = np.random.permutation(len(sequences))
sequences = sequences[shuffledIndices]
labels = label[shuffledIndices]

trainSize = int(len(sequences) * 0.8)

trainSequences = sequences[:trainSize]
trainLabels = labels[:trainSize]

otherSequences = sequences[trainSize:]
otherLabels = labels[trainSize:]

shuffledIndices = np.random.permutation(len(otherSequences))
otherSequences = otherSequences[shuffledIndices]
otherLabels = otherLabels[shuffledIndices]

validationSize = int(len(otherSequences) * 0.5)

validationSequences = otherSequences[:validationSize]
validationLabels = otherLabels[:validationSize]

testSequences = otherSequences[validationSize:]
testLabels = otherLabels[validationSize:]

#print(trainSequences)
#print(validationSequences)
#print(testSequences)

def encoder(inputs, headSize, numberHeads, feedForwardDimention, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=headSize, num_heads=numberHeads, dropout=dropout)(x, x)
    x = Add()([x, inputs])

    # Feed Forward Part
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(feedForwardDimention, activation='relu')(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)# 映射回原始维度
    return Add()([y, x])

def transformer(inputShape, headSize, numberHeads, feedForwardDimention, numberLayers, dropout=0):
    inputs = Input(shape=inputShape)
    x = inputs
    for _ in range(numberLayers):
        x = encoder(x, headSize, numberHeads, feedForwardDimention, dropout)
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    outputs = Dense(1, activation='linear')(x)
    return Model(inputs=inputs, outputs=outputs)

inputShape = trainSequences.shape[1:]
headSize = 256
numberHeads = 16
feedForwardDimention = 1024
numberLayers = 12
dropout = 0.20

model = transformer(inputShape, headSize, numberHeads, feedForwardDimention, numberLayers, dropout)


def custom_mae_loss(y_true, y_pred): # MAE(Mean Absolute Error)损失函数
    y_true_next = tf.cast(y_true[:, 1], tf.float64)  # tf.cast(): 将一个张量的数据类型进行转换 -> 将y_true中第二列的数据提取并转换为tf.float64
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)  
    abs_error = tf.abs(y_true_next - y_pred_next)  # 计算mae, abs_error的返回值是一个一维张量
    return tf.reduce_mean(abs_error)  # tf.reduce_mean(): 计算平均值, 参数是张量, 返回值是标量

def custom_wmape_loss(y_true, y_pred):  
    y_true_next = tf.cast(y_true[:, 1], tf.float64)  
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)  
    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)  
    y_true_next_unscaled = (y_true_next * std) + mean  
    y_pred_next_unscaled = (y_pred_next * std) + mean  
    abs_error = tf.abs(y_true_next_unscaled - y_pred_next_unscaled)  
    relative_error = abs_error / (tf.abs(y_true_next_unscaled) + tf.ones_like(y_true_next_unscaled) * 1e-8)  
    weighted_error = abs_error * relative_error  
    return tf.reduce_mean(weighted_error)

#def dir_acc(y_true, y_pred):
#    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)  # 反归一化价格, 这里的y_true[:, 2]对应于sequence的第三行
#    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean  # Un-scale previous true price
#    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean  # sequence的第二行是ClosePrice
#    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean  # Un-scale predicted next price
#
#    true_change = y_true_next - y_true_prev  # 真实价格变化
#    pred_change = y_pred_next - y_true_prev  # 预测价格变化
#
#    correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))  # sign()提取输入张量的符号, 如果输入值>0, 返回1.0; 输入值<0, 返回-1.0 
#    # equal()比较是否相等, 相等返回1.0, 不相等返回0.0
#    return tf.reduce_mean(tf.cast(correct_direction, tf.float64))  # 

def dir_acc(y_true, y_pred, threshold=0.005):  
    # 保持原有反归一化逻辑  
    mean = tf.cast(y_true[:, 2], tf.float64)  
    std = tf.cast(y_true[:, 3], tf.float64)  
    
    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean  
    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean  
    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean  
    
    true_change = y_true_next - y_true_prev  
    pred_change = y_pred_next - y_true_prev  
    
    # 动态波动阈值  
    min_fluctuation = threshold * y_true_prev  
    is_significant = tf.abs(true_change) >= min_fluctuation  
    
    # 使用掩码过滤无效波动  
    valid_true_sign = tf.sign(tf.boolean_mask(true_change, is_significant))  
    valid_pred_sign = tf.sign(tf.boolean_mask(pred_change, is_significant)) # sign()提取输入张量的符号, 如果输入值>0, 返回1.0; 输入值<0, 返回-1.0 
    # equal()比较是否相等, 相等返回1.0, 不相等返回0.0 
    
    # 核心修改点：统一数据类型  
    correct = tf.equal(valid_true_sign, valid_pred_sign)  
    
    # 解决方案1：强制转换返回类型  
    return tf.cond(  
        tf.size(correct) > 0,  
        lambda: tf.reduce_mean(tf.cast(correct, tf.float64)),  
        lambda: tf.constant(0.0, dtype=tf.float64)  # 统一使用float64  
    )  

    # 解决方案2：使用安全系数计算  
    # safe_count = tf.maximum(tf.cast(tf.size(correct), tf.float64), 1e-7)  
    # return tf.reduce_sum(tf.cast(correct, tf.float64)) / safe_count

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "transformer_train_model_wmape.keras",  # Filepath to save the best model
    monitor="dir_acc",  #"loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=0,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "transformer_val_model_wmape.keras",  # Filepath to save the best model
    monitor="val_dir_acc", #"val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=0,  # Display progress
)

def get_lr_callback(batch_size=16, mode='cos', epochs=500, plot=False):
    lr_start, lr_max, lr_min = 0.0001, 0.005, 0.00001  # Adjust learning rate boundaries
    lr_ramp_ep = int(0.30 * epochs)  # 30% of epochs for warm-up
    lr_sus_ep = max(0, int(0.10 * epochs) - lr_ramp_ep)  # Optional sustain phase, adjust as needed

    def lrfn(epoch):
        if epoch < lr_ramp_ep:  # Warm-up phase
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:  # Sustain phase at max learning rate
            lr = lr_max
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            lr = lr_min  # Default to minimum learning rate if mode is not recognized

        return lr

    if plot:  # 学习率变化曲线 plot默认为False
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

BATCH_SIZE = 128  # Number of training examples used to calculate each iteration's gradient
EPOCHS = 100  # Total number of times the entire dataset is passed through the network

model.compile(
    optimizer = 'adam', # 优化器
    loss = custom_wmape_loss, # 损失函数
    metrics = [dir_acc] # 评估指标
)

model.fit(
    trainSequences,  # Training features
    trainLabels,  # Training labels
    validation_data=(validationSequences, validationLabels),  # Validation data
    epochs=EPOCHS,  # Number of epochs td o train for
    batch_size=BATCH_SIZE,  # Size of each batch
    verbose=0,
    shuffle=True,  # Shuffle training data before each epoch
    callbacks=[checkpoint_callback_train, checkpoint_callback_val, get_lr_callback(batch_size=BATCH_SIZE, epochs=EPOCHS, plot=False)]  # Callbacks for saving models and adjusting learning rate
)

model.load_weights("transformer_train_model_wmape.keras")  # Load the best model from the validation phase
accuracy_train = model.evaluate(testSequences, testLabels)[1]  # Evaluate the model on the test data
print('train accuracy: ' + str(accuracy_train))

model.load_weights("transformer_val_model_wmape.keras")  # Load the best model from the validation phase
accuracy_validation = model.evaluate(testSequences, testLabels)[1]  # Evaluate the model on the test data
print('validation accuracy: ' + str(accuracy_validation))

# 加载在验证集上最优的模型权重
model.load_weights("transformer_val_model_wmape.keras")

# 1. 预测并反归一化
# 预测（归一化后的价格）
preds_norm = model.predict(testSequences)

# 真实值（归一化后的下一时刻价格）
actuals_norm = testLabels[:, 1]

# 反归一化： price = norm * std + mean
mean_price = stats['ClosePrice_mean'].values[0]
std_price  = stats['ClosePrice_std'].values[0]

preds = preds_norm.flatten() * std_price + mean_price
actuals = actuals_norm       * std_price + mean_price

# 2. 绘制预测效果图
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Close Price')
plt.plot(preds,   label='Predicted Close Price')
plt.title('Actual vs. Predicted Close Price')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show(block=True)

model.summary()
# 3. 统计量计算
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(actuals, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE:  {mae:.4f}')
print(f'R2:   {r2:.4f}')
