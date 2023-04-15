import numpy as np
from sklearn.metrics import r2_score




def cal_l2_relative_error(pred, true):
    err = np.linalg.norm(pred - true) / np.linalg.norm(true)
    return err

def cal_mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def cal_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def cal_rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
                                        
def cal_r2(y_pred, y_true):
    return r2_score(y_true, y_pred)

def cal_metrics(pred, true, on_matrix=False):
    # on_matrix: 如果为真，则在整个矩阵上计算误差，否则在每一列上计算误差
    if on_matrix:
        l2 = cal_l2_relative_error(pred, true)
        mae = cal_mae(pred, true)
        mse = cal_mse(pred, true)
        rmse = cal_rmse(pred, true)
        r2 = cal_r2(pred, true)
    else:
        l2 = [cal_l2_relative_error(pred[:, i], true[:, i]) for i in range(pred.shape[1])]
        mae = [cal_mae(pred[:, i], true[:, i]) for i in range(pred.shape[1])]
        mse = [cal_mse(pred[:, i], true[:, i]) for i in range(pred.shape[1])]
        rmse = [cal_rmse(pred[:, i], true[:, i]) for i in range(pred.shape[1])]
        r2 = [cal_r2(pred[:, i], true[:, i]) for i in range(pred.shape[1])]
    return l2, mae, mse, rmse, r2