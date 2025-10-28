import numpy as np
from sklearn.metrics import mean_absolute_error

def mae(y, yhat):  return float(mean_absolute_error(y, yhat))
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat)/np.clip(np.abs(y), 1e-6, None)))*100)
