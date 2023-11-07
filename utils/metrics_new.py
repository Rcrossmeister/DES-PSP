import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryMatthewsCorrCoef
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

# 分类指标计算函数
def classification_metrics(y_pred, y_true):
    # 实例化指标
    accuracy = BinaryAccuracy()
    f1 = BinaryF1Score()
    mcc = BinaryMatthewsCorrCoef()
    
    # 计算指标
    acc = accuracy(y_pred, y_true)
    f1_score = f1(y_pred, y_true)
    mcc_score = mcc(y_pred, y_true)
    
    return acc, f1_score, mcc_score

# 回归指标计算函数
def regression_metrics(y_pred, y_true):
    # 实例化指标
    mse = MeanSquaredError()
    rmse = MeanSquaredError(squared=False)
    mae = MeanAbsoluteError()
    
    # 计算指标
    mse_score = mse(y_pred, y_true)
    rmse_score = rmse(y_pred, y_true)
    mae_score = mae(y_pred, y_true)
    
    # 计算ADE和FDE（需要自定义实现，这里只是一个占位函数）
    ade = calculate_ade(y_pred, y_true)  # 自定义函数计算ADE
    fde = calculate_fde(y_pred, y_true)  # 自定义函数计算FDE
    
    return mse_score, rmse_score, mae_score, ade, fde

# 自定义ADE和FDE计算
# 定义计算ADE和FDE的函数
def calculate_ade(y_pred, y_true):
    """
    计算平均位移误差（ADE）

    参数:
    y_pred (torch.Tensor): 预测轨迹点，形状为 (batch_size, sequence_length, num_dimensions)
    y_true (torch.Tensor): 真实轨迹点，形状为 (batch_size, sequence_length, num_dimensions)

    返回:
    torch.Tensor: ADE值
    """
    ade = torch.norm(y_pred - y_true, p=2, dim=-1).mean()
    return ade

def calculate_fde(y_pred, y_true):
    """
    计算最终位移误差（FDE）

    参数:
    y_pred (torch.Tensor): 预测轨迹点，形状为 (batch_size, sequence_length, num_dimensions)
    y_true (torch.Tensor): 真实轨迹点，形状为 (batch_size, sequence_length, num_dimensions)

    返回:
    torch.Tensor: FDE值
    """
    fde = torch.norm(y_pred[:, -1, :] - y_true[:, -1, :], p=2, dim=-1).mean()
    return fde
