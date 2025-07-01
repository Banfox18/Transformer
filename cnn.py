import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =================== 数据读取 ===================
ticker = "000300"  # 示例股票代码，可替换
data = pd.read_csv('Data/CSI 300/' + ticker + '.csv')

# 提取特征
closePrice = data['ClosePrice']
data['Date'] = pd.to_datetime(data['CloseDate'])
data.set_index("Date", inplace=True)
closePrice = data['ClosePrice']

# =================== 数据切分 ===================
# 常见做法：80%训练集，20%测试集（时序，不可打乱）
n_total = len(closePrice)
n_train = int(0.8 * n_total)
train, test = closePrice[:n_train], closePrice[n_train:]

# =================== ARIMA建模与预测 ===================
# 选择定阶，这里直接用(5,1,0)为例，可用AIC进行阶数调优
order = (5, 1, 0)
model = ARIMA(train, order=order)
model_fit = model.fit()

# 在测试集上预测
start = test.index[0]
end = test.index[-1]
pred = model_fit.predict(start=start, end=end, typ='levels')

# =================== 评估指标 ===================
y_test_real = test
y_pred_real = pred

mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print(f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}")

# =================== 可视化 ===================
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, y_pred_real, label='Pred', linestyle='dashed')
plt.legend()
plt.title(f'ARIMA({order}) Stock Closing Price Prediction')
plt.show()