import numpy as np
import scipy

def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    return scipy.optimize.approx_fprime(w, function, eps)
    if len(w.shape) == 1:
        result = np.zeros(w.shape[0])
        f = function(w)
        for i in range(0, w.shape[0]):
            e_i = np.zeros(w.shape[0])
            e_i[i] = 1
            result[i] = (function(w + eps * e_i) - f) / eps
    else:
        result = np.zeros(w.shape)
        for i in range(0, w.shape[0]):
            f = function(w[i])
            for j in range(0, w.shape[1]):
                e_j = np.zeros(w.shape[1])
                e_j[j] = 1
                result[i][j] = (function(w[i] + eps * e_j) - f) / eps
    return result