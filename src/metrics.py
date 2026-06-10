"""
MedMLP 评估指标
================
纯 NumPy 实现分类评估指标：Accuracy、Macro F1-Score、Confusion Matrix。
"""

import numpy as np


def accuracy(y_pred, y_true):
    """
    计算分类准确率。
    
    Parameters
    ----------
    y_pred : ndarray, shape (N,)
        预测类别索引
    y_true : ndarray, shape (N,)
        真实类别索引
        
    Returns
    -------
    acc : float
        准确率 [0, 1]
    """
    return np.mean(y_pred == y_true)


def confusion_matrix(y_pred, y_true, num_classes):
    """
    计算混淆矩阵（矩阵化统计）。
    
    Parameters
    ----------
    y_pred : ndarray, shape (N,)
    y_true : ndarray, shape (N,)
    num_classes : int
    
    Returns
    -------
    cm : ndarray, shape (num_classes, num_classes)
        cm[i, j] = 真实类为 i 且预测类为 j 的样本数
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_score(y_pred, y_true, num_classes):
    """
    计算多分类 Macro F1-Score。
    
    对每个类别分别计算 Precision、Recall、F1，然后取简单平均。
    
    Parameters
    ----------
    y_pred : ndarray, shape (N,)
    y_true : ndarray, shape (N,)
    num_classes : int
    
    Returns
    -------
    f1 : float
        Macro F1-Score [0, 1]
    """
    cm = confusion_matrix(y_pred, y_true, num_classes)
    
    f1_scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        # 精确率：TP / (TP + FP)，FP = 该列之和 - TP
        fp = np.sum(cm[:, c]) - tp
        # 召回率：TP / (TP + FN)，FN = 该行之和 - TP
        fn = np.sum(cm[c, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)
