
# Transformer股价预测模型

本项目使用**Transformer神经网络**对股票（如沪深300）收盘价进行序列建模和未来10日价格预测。模型利用技术指标（布林带宽度、RSI、ROC等）作为输入特征，通过自定义损失函数和自定义方向准确率指标进行优化和评价。

## 主要特性

- **数据处理**：支持布林带、RSI、ROC等常见技术指标自动提取。
- **特征工程**：多种模型输入特征与标准化处理。
- **建模架构**：采用多层Transformer编码器结构，带多头自注意力和前馈神经网络。
- **自定义损失与评价**：WMAPE损失、方向准确率评价。
- **训练与验证**：包含模型保存、学习率调度与测试评估流程。
- **可视化**：支持实际值与预测值绘制、关键统计量输出。

## 环境依赖

- Python 3.x
- pandas
- numpy
- tensorflow (推荐2.11以上)
- scikit-learn
- matplotlib

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## 数据准备

- 数据路径：`Data/CSI 300/daily.csv`
- 必须包含字段：`OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, ChangeRatio, CloseDate`

## 项目流程简述

1. **计算技术指标**：调用`calculate_bollinger_bands`、`calculate_rsi`、`calculate_roc`对收盘价计算特征。
2. **数据标准化**：计算均值和标准差，对所有输入特征进行标准化。
3. **序列切片**：构造历史`SEQUENCE_LEN`步窗口输入和未来`PREDICTION_LEN`步标签，按80%/10%/10%切分数据。
4. **模型结构**：`transformer()`方法定义堆叠多层的编码器，每层含自注意力、前馈和残差结构。
5. **自定义损失与评价**：包括WMAPE损失和方向准确率`dir_acc`（可识别显著波动）。
6. **训练与保存**：通过回调自动保存表现最优参数，利用学习率调度器优化训练过程。
7. **测试与可视化**：输出预测与实际对比图和MSE、RMSE、MAE、R²等统计指标。

## 主要自定义函数说明

- **calculate_bollinger_bands**：计算布林带上下轨。
- **calculate_rsi**：计算相对强弱指数。
- **calculate_roc**：计算变化率（Rate of Change）。
- **custom_wmape_loss**：加权绝对误差损失，反映归一化误差。
- **dir_acc**：按阈值评估预测涨跌方向准确率。

## 实验结果展示

训练与验证完成后，可以在终端看到：
- 训练集和验证集方向准确率
- 画出预测和实际收盘价曲线
- 输出MSE、RMSE、MAE、R2等指标

## 运行方法

通常直接运行`main.ipynb`即可，但由于`TensorFlow 2.11`与Jupyter Kernal部分版本之间存在不兼容的问题导致Kernel崩溃，此时可运行`main.py`.

确认csv数据存在，并运行主脚本：

```bash
python main.py
```
或者复制本README里的全部代码运行。


## 结果图示

运行后将弹出实际vs预测收盘价对比图。

---

**如需自定义窗口长度、特征种类或模型层数，请直接修改对应参数。欢迎二次开发！**
