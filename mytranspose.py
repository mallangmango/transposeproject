import numpy as np
import pandas as pd
import torch

def mytranspose(x):
    if isinstance(x, np.ndarray):
        return x.T
    elif isinstance(x, pd.DataFrame):
        return x.transpose()
    elif torch.is_tensor(x):
        return x.T
    else:
        raise TypeError("지원되지 않는 타입입니다.")
