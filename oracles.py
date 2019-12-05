import numpy as np
import scipy.special
import scipy.sparse

class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """

        return (np.mean(np.logaddexp(0, - X.dot(w) * y)) +
                0.5 * self.l2_coef * np.dot(w, w))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        if isinstance(X, scipy.sparse.csr_matrix):
            grad = (-np.mean(X.multiply((y * scipy.special.expit(-X.dot(w) * y))
                [:, np.newaxis]), axis=0) + self.l2_coef * w)
            return np.ravel(grad)
        return (-np.mean(X * (y * scipy.special.expit(-X.dot(w) * y))
                [:, np.newaxis], axis=0) + self.l2_coef * w)
