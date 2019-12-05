import numpy as np
import oracles
import time
import scipy.special


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta - float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить
        оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних
        значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        if loss_function == 'binary_logistic':
            self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])

    def fit(self, X, y, w_0=None, trace=False, test=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий
        информацию о поведении метода. Длина словаря history = количество
        итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя
        итерациями метода
        history['func']: list of floats, содержит значения функции на каждой
        итерации(0 для самой первой точки)
        """
        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                w_0 = np.zeros(X.shape[1])
        self.w = w_0
        prev_func = self.get_objective(X, y)
        k = 0
        if trace:
            history = {'time': [], 'func': []}
            history['time'].append(0)
            history['func'].append(self.get_objective(X, y))
            start_time = time.perf_counter()
        while True:
            k += 1
            step = self.step_alpha / (k ** self.step_beta)
            self.w = self.w - step * self.get_gradient(X, y)
            if trace:
                history['time'].append(time.perf_counter() - start_time)
                history['func'].append(self.get_objective(X, y))
            func = self.get_objective(X, y)
            if (k >= self.max_iter) or \
                    (abs(func - prev_func) < self.tolerance):
                break
            prev_func = func
        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        if self.w is None:
            raise Exception('Not trained yet')
        if self.loss_function == 'binary_logistic':
            return np.sign(X.dot(self.w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует
        вероятности принадлежности i-го объекта к классу k
        """
        if self.loss_function == 'binary_logistic':
            return scipy.special.expit(X.dot(self.w))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        if self.w is None:
            raise Exception('Not trained yet')
        else:
            return self.w

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / y.shape[0]


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить
        оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних
        значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать
        np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = None
        if loss_function == 'binary_logistic':
            self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, test=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий
        информацию о поведении метода. Если обновлять history после каждой
        итерации, метод перестанет превосходить в скорости метод GD. Поэтому,
        необходимо обновлять историю метода лишь после некоторого числа
        обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
        {кол-во объектов, обработанных SGD} / {кол-во объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя
        значениями приближённого номера эпохи будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет
        записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя
        соседними замерами
        history['func']: list of floats, содержит значения функции после
        текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы
        разности векторов весов с соседних замеров (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                w_0 = np.zeros(X.shape[1])
        self.w = w_0
        k = 0
        if trace:
            history = {'epoch_num': [], 'time': [], 'func': [], 'weights_diff':
                       []}
            history['epoch_num'].append(0)
            history['time'].append(0)
            history['func'].append(self.get_objective(X, y))
            history['weights_diff'].append(0)
            prev_epoch_num = 0
            start_time = time.perf_counter()
            prev_epoch_w = 0
        ind = np.random.permutation(X.shape[0])
        batch = 0
        prev_func = self.get_objective(X, y)
        while True:
            k += 1
            if (batch >= X.shape[0]):
                np.random.shuffle(ind)
                batch = 0
            step = self.step_alpha / (k ** self.step_beta)
            self.w = (self.w - step * self.get_gradient(X[ind[batch: batch +
                      self.batch_size]],
                      y[ind[batch: batch + self.batch_size]]))
            if trace:
                epoch_num = k * self.batch_size / X.shape[0]
                if epoch_num - prev_epoch_num >= log_freq:
                    history['epoch_num'].append(epoch_num)
                    history['time'].append(time.perf_counter() - start_time)
                    history['func'].append(self.get_objective(X, y))
                    history['weights_diff'].append(np.linalg.norm(self.w -
                                                   prev_epoch_w))
                    prev_epoch_w = self.w
                    prev_epoch_num = epoch_num

            batch += self.batch_size
            func = self.get_objective(X, y)
            if (k >= self.max_iter) or \
                    (np.linalg.norm(func - prev_func) < self.tolerance):
                break
            prev_func = func
        if trace:
            return history
