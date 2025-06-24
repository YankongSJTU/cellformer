import torch
import numpy as np
from lifelines.utils import concordance_index

def calculate_cindexbackup(risks, times, events):
    """Calculate concordance index for survival predictions."""
    return concordance_index(times, -risks, events)

def regularize_weights(model, norm=1):
    """Calculate L1 regularization term for model weights."""
    reg = 0
    for param in model.parameters():
        reg += torch.norm(param, p=norm)
    return reg
def calculate_cindex(risks, times, events):
    """计算C-index (concordance index)，处理NaN值"""
    # 转换为numpy数组
    risks = np.asarray(risks)
    times = np.asarray(times)
    events = np.asarray(events)
    
    # 检查并移除NaN值
    mask = ~(np.isnan(risks) | np.isnan(times) | np.isnan(events))
    risks = risks[mask]
    times = times[mask]
    events = events[mask]
    
    if len(risks) == 0:
        return 0.5  # 默认值
    
    from lifelines.utils import concordance_index
    return concordance_index(times, -risks, events)
