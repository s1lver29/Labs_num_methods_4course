import numpy as np


def method_newton(X, f: np.ndarray, w: np.ndarray, eps: float = 1e-3, max_iter=100) -> np.ndarray:
    """
    Вычисление системы нелиныйных арифметических уравнений методом Ньютона (Улучшение метода Ньютона)
    :params: X:list -- примерное приближение для каждой переменной в уравнении
             f:np.ndarray --  система уравнений F(x) = 0
             W:np.ndarray -- матрица якоби W(x) для системы уравнений F(x)
             eps:float=1e-5 -- точность вычисления
             max_iter=100 -- максимальное количество итераций
    :return: np.ndarray -- решение системы уравнений 
    """
    x = np.array(X)
    iters = {} # Словарь всех итераций
    iters[0] = x.copy()
    iter = 1

    while iter <= max_iter:
        F, W = f(x[0], x[1])*(-1), w(x[0], x[1])

        delta_x = np.linalg.solve(W, F)
        x += delta_x

        iters[iter] = x.copy()
        if max(abs(delta_x)) < eps:
            return iters
        iter += 1


def fin_diff(a:float, b:float, h:float, p_x, q_x, f_x, first_equation:list, last_equation:list) -> list:
    """
    Вычисление обыкновенного дифференциального уравнения второго порядка конечно-разностным методом
    :params: a:float -- левая граница
             b:float -- правая граница
             h:float -- шаг сетки
             p_x:func -- коэффициент перед y'
             q_x:func -- коэффициент перед y
             f_x:func -- свободный коэффициент
             first_equation:list -- коэффициенты первого уравнения аппроксимации производных первого порядка
             last_equation:list -- коэффициенты последнего уравнения аппроксимации производных первого порядка
    :return: tuple(list, list) -- значения по оси x и y
    """
    N = int((b - a)/h + 1)

    matrix = np.zeros((N, N+1))
    for i, j in enumerate([0, 1, N]): matrix[0, j] = first_equation[i]
    for i, j in enumerate([-3,-2,-1]): matrix[N-1, j] = last_equation[i]

    x_arr = np.arange(a, b + h, h)

    for i in range(1, N-1):
        matrix[i][i-1] = 1 - p_x(x_arr[i])*h/2
        matrix[i][i] = -2 + h**2*q_x(x_arr[i])
        matrix[i][i+1] = 1 + p_x(x_arr[i])*h/2
        matrix[i][N] = h*h*f_x(x_arr[i])
    
    y_arr = np.linalg.solve(matrix[:,:-1], matrix[:,-1])
    return x_arr, y_arr 


if __name__ == "__main__":
    # Решения СНАУ методом Ньютона
    def f(p, q): return np.array(
        [np.sin(p + .5) + q, np.cos(q - .5) - 1.1*p - 3])
    def w(p, q): return np.array([[np.cos(p + .5), 1], [-1.1, -np.sin(q - .5)]])
    X = [i/2 for i in [-2, 2]]
    iters = method_newton(X, f, w)

    # Решение ОДУ конечно-разностным методом
    a, b, h = 1, 2, 0.02
    p_x = lambda x: -(2*x+1)/x
    q_x = lambda x: (x+1)/x
    f_x = lambda x: 0
    axis_x, axis_y = fin_diff(a, b, h, p_x, q_x, f_x, [-1, 1, 3*np.exp(1)*h], [1, (2*h-1), 0])
