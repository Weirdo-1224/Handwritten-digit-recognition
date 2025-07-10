# coding: utf-8
import numpy as np

import numpy as np

# 计算一维函数的数值梯度
def _numerical_gradient_1d(f, x):
    h = 1e-4  # 微小的差值，通常为0.0001
    grad = np.zeros_like(x)  # 创建一个和输入x形状相同的全零数组，用于存储梯度

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 计算 f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # 计算 f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 数值梯度的计算

        x[idx] = tmp_val  # 还原值

    return grad

# 计算二维函数的数值梯度
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)  # 创建一个和输入X形状相同的全零数组，用于存储梯度

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad

# 通用的数值梯度计算函数，支持多维输入
def numerical_gradient(f, x):
    h = 1e-4  # 微小的差值，通常为0.0001
    grad = np.zeros_like(x)  # 创建一个和输入x形状相同的全零数组，用于存储梯度

    # 创建迭代器，用于遍历x的所有元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 计算 f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # 计算 f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 数值梯度的计算

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad
