
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import interp1d

def find_poincare_points(sol:OdeResult):
    phi = sol.y[3, :]
    t = sol.t

    # 1. Определяем границы и создаем цели
    phi_min, phi_max = min(phi[0], phi[-1]), max(phi[0], phi[-1])
    # Ищем все целые 2*pi внутри диапазона
    start_k = np.ceil(phi_min / (2 * np.pi))
    end_k = np.floor(phi_max / (2 * np.pi))
    
    phi_targets = np.arange(start_k, end_k + 1) * (2 * np.pi)
    # Оставляем только те, что внутри (исключая границы, если нужно)
    phi_targets = phi_targets[(phi_targets > phi_min) & (phi_targets < phi_max)]

    if phi_targets.size == 0:
        print("phi_targets is empty")
        sys.exit(1)
        #return np.empty((sol.y.shape[0], 0))

    # 2. Переворачиваем данные, если phi убывает, чтобы аргумент всегда рос
    if phi[-1] < phi[0]:
        phi_for_interp = phi[::-1]
        t_for_interp = t[::-1]
    else:
        phi_for_interp = phi
        t_for_interp = t

    # 3. Интерполяция и получение координат
    t_of_phi = interp1d(phi_for_interp, t_for_interp, kind='cubic')
    t_targets = t_of_phi(phi_targets)

    # Получаем координаты в эти моменты
    poincare_data = sol.sol(t_targets)
    df = pd.DataFrame(poincare_data.T, columns=['ppar','r','theta','phi'])
    df['tau'] =  t_targets
    return df