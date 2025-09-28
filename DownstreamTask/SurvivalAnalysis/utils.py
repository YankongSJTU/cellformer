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
    """ C-index (concordance index)， """
    #  
    risks = np.asarray(risks)
    times = np.asarray(times)
    events = np.asarray(events)
    
    # remove NaN 
    mask = ~(np.isnan(risks) | np.isnan(times) | np.isnan(events))
    risks = risks[mask]
    times = times[mask]
    events = events[mask]
    
    if len(risks) == 0:
        return 0.5  # default
    from lifelines.utils import concordance_index
    return concordance_index(times, -risks, events)
