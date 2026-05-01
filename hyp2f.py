from numba import njit
import scipy
import numpy as np
@njit
def fast_hyp_part(x, n, terms=10):
    """
    Аппроксимация hyp2f1(0.5, (2+n)/2, (4+n)/2, x^2) / (2+n)
    через степенной ряд.
    """
    x2 = x**2
    a = 0.5
    b = (2.0 + n) / 2.0
    c = (4.0 + n) / 2.0
    
    # Первый член ряда (k=0) всегда 1
    hyp_sum = 1.0
    current_term = 1.0
    
    # Итерируемся для точности (10-15 итераций обычно за глаза при x < 0.9)
    for k in range(1, terms):
        # Рекуррентное отношение для следующего члена ряда
        # (a+k-1)*(b+k-1) / ((c+k-1)*k) * z
        multiplier = ((a + k - 1) * (b + k - 1)) / ((c + k - 1) * k) * x2
        current_term *= multiplier
        hyp_sum += current_term
        
        # Если член ряда стал ничтожно мал, выходим раньше
        if abs(current_term) < 1e-12:
            break
            
    return hyp_sum

@njit
def fast_hyp2f1_specific(x, n, terms=15):
    """
    Аппроксимация hyp2f1(0.5, (1+n)/2, (3+n)/2, x^2)
    через рекуррентный расчет степенного ряда.
    """
    x2 = x**2
    a = 0.5
    b = (1.0 + n) / 2.0
    c = (3.0 + n) / 2.0
    
    hyp_sum = 1.0
    current_term = 1.0
    
    for k in range(1, terms):
        # Формула: term_{k} = term_{k-1} * (a+k-1)*(b+k-1) / ((c+k-1)*k) * z
        multiplier = ((a + k - 1) * (b + k - 1)) / ((c + k - 1) * k) * x2
        current_term *= multiplier
        hyp_sum += current_term
        
        if abs(current_term) < 1e-14:
            break
            
    return hyp_sum




def create_fast_hyp(a, b, c, n_points=1000):
    # 1. Считаем таблицу один раз при создании функции
    z_table = np.linspace(0, 1.0, n_points).astype(np.float64)
    f_table = scipy.special.hyp2f1(a, b, c, z_table).astype(np.float64)
    
    # Шаг сетки для мгновенного поиска индекса
    dz = 1.0 / (n_points - 1)

    # 2. Создаем саму njit-функцию, которая «захватит» таблицы
    @njit
    def nested_hyp(z):
        # Ограничения, чтобы не выйти за пределы массива
        if z <= 0.0: return f_table[0]
        if z >= 1.0: return f_table[-1]
        
        # Индекс в равномерной сетке за O(1)
        idx = int(z / dz)
        
        # Линейная интерполяция
        z0 = idx * dz
        y0 = f_table[idx]
        y1 = f_table[idx + 1]
        
        return y0 + (y1 - y0) * (z - z0) / dz

    return nested_hyp


def create_pade_hyp(a, b, c, order=2):
    """
    Создает быструю функцию на основе аппроксимации Паде.
    order: степень полиномов (обычно 5-8 достаточно для высокой точности)
    """
    # 1. Генерируем коэффициенты ряда Тейлора для hyp2f1
    # Нам нужно (2 * order + 1) коэффициентов для Паде
    n_coeffs = 2 * order + 1
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0
    curr = 1.0
    for n in range(1, n_coeffs):
        curr *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n)
        coeffs[n] = curr
    
    # 2. Вычисляем коэффициенты полиномов Паде (P/Q)
    # pade возвращает два объекта poly1d, берем их коэффициенты в обратном порядке (от младшей степени)
    p_poly, q_poly = scipy.interpolate.pade(coeffs, order)
    p_coeffs = np.array(p_poly.coeffs)[::-1]
    q_coeffs = np.array(q_poly.coeffs)[::-1]

    # 3. Компилируем быструю функцию через Numba
    @njit
    def fast_pade(z):
        # Схема Горнера для быстрого вычисления полиномов
        p_val = 0.0
        for i in range(len(p_coeffs)-1, -1, -1):
            p_val = p_val * z + p_coeffs[i]
            
        q_val = 0.0
        for i in range(len(q_coeffs)-1, -1, -1):
            q_val = q_val * z + q_coeffs[i]
            
        return p_val / q_val

    return fast_pade