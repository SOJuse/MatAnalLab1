import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from math import sin, cos

# --- Исходная функция ---
def f(x):
    return -np.cos(2 * x)

# --- Точное значение интеграла ---
def true_integral():
    return -0.5 * sin(2)

# --- Метод прямоугольников ---
def rectangle_method(f: Callable, a: float, b: float, n: int, mode='mid') -> float:
    h = (b - a) / n
    result = 0
    for i in range(n):
        if mode == 'left':
            x = a + i * h
        elif mode == 'right':
            x = a + (i + 1) * h
        elif mode == 'mid':
            x = a + (i + 0.5) * h
        elif mode == 'random':
            x = a + i * h + np.random.uniform(0, h)
        else:
            raise ValueError("Неверный режим: 'left', 'right', 'mid', 'random'")
        result += f(x)
    return result * h

# --- Метод трапеций ---
def trapezoidal_method(f: Callable, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

# --- Метод Симпсона ---
def simpson_method(f: Callable, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Метод Симпсона требует чётное количество отрезков n")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

# --- Построение графиков для метода ---
def plot_method(f, a, b, n, method_name, method_fn):
    x_vals = np.linspace(a, b, 1000)
    y_vals = f(x_vals)
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x)', color='black')

    h = (b - a) / n
    for i in range(n):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        if method_name == 'left':
            y = f(x0)
            plt.bar(x0, y, width=h, align='edge', alpha=0.4)
        elif method_name == 'right':
            y = f(x1)
            plt.bar(x0, y, width=h, align='edge', alpha=0.4)
        elif method_name == 'mid':
            x_mid = (x0 + x1) / 2
            y = f(x_mid)
            plt.bar(x0, y, width=h, align='edge', alpha=0.4)
        elif method_name == 'trapezoid':
            xs = [x0, x1]
            ys = [f(x0), f(x1)]
            plt.fill_between(xs, ys, alpha=0.4)
        elif method_name == 'simpson':
            xs = np.linspace(x0, x1, 100)
            ys = f(xs)
            plt.fill_between(xs, ys, alpha=0.4)

    plt.title(f"Метод: {method_name.title()}, n={n}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Анализ ошибок ---
def error_analysis():
    exact = true_integral()
    ns = [2**i for i in range(0, 8)]
    results = {
        'midpoint': [],
        'trapezoid': [],
        'simpson': []
    }
    for n in ns:
        rect = rectangle_method(f, 0, 1, n, 'mid')
        trap = trapezoidal_method(f, 0, 1, n)
        simp = simpson_method(f, 0, 1, n if n % 2 == 0 else n + 1)
        results['midpoint'].append(abs(rect - exact))
        results['trapezoid'].append(abs(trap - exact))
        results['simpson'].append(abs(simp - exact))

    plt.figure(figsize=(8, 5))
    for label, errors in results.items():
        plt.plot(ns, errors, marker='o', label=label)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Число отрезков n')
    plt.ylabel('Абсолютная ошибка')
    plt.title('График ошибки по мере увеличения n')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

# --- Основной запуск ---
if __name__ == "__main__":
    a, b = 0, 1
    n = 8
    print("Точное значение интеграла:", true_integral())

    print("\n--- Метод прямоугольников (середина) ---")
    print(rectangle_method(f, a, b, n, 'mid'))

    print("\n--- Метод трапеций ---")
    print(trapezoidal_method(f, a, b, n))

    print("\n--- Метод Симпсона ---")
    print(simpson_method(f, a, b, n if n % 2 == 0 else n + 1))

    for n in [4, 8, 16]:
        plot_method(f, a, b, n, 'mid', rectangle_method)
        plot_method(f, a, b, n, 'trapezoid', trapezoidal_method)
        plot_method(f, a, b, n, 'simpson', simpson_method)

    error_analysis()