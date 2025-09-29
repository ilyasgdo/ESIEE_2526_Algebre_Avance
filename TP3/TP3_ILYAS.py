import numpy as np
import matplotlib.pyplot as plt


def Base(X, k, x):
    """Base de Newton N_k(x) = prod_{j=0..k-1} (x - X[j]).
    Accepte scalar ou ndarray pour x.
    """
    X = np.asarray(X, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if k == 0:
        return np.ones_like(x_arr) if x_arr.ndim else 1.0
    p = np.ones_like(x_arr, dtype=float)
    for j in range(k):
        p = p * (x_arr - X[j])
    return p if x_arr.ndim else float(p)

def compute_with_diff_divise(X, Y):
    """Calcule les coefficients (différences divisées) en place.
    Retourne c tel que P(x)=c[0] + c[1](x-x0)+c[2](x-x0)(x-x1)+...
    """
    X = np.asarray(X, dtype=float)
    c = np.asarray(Y, dtype=float).copy()
    n = len(X)
    for order in range(1, n):
        for i in range(n-1, order-1, -1):
            c[i] = (c[i] - c[i-1]) / (X[i] - X[i-order])
    return c

def eval_newton(X, Y, x):
    """Évalue le polynôme de Newton en x (scalaire ou vecteur) via Horner adapté."""
    X = np.asarray(X, dtype=float)
    coeff = compute_with_diff_divise(X, Y)
    x_arr = np.atleast_1d(x).astype(float)
    res = np.full_like(x_arr, coeff[-1], dtype=float)
    for k in range(len(coeff)-2, -1, -1):
        res = res * (x_arr - X[k]) + coeff[k]
    return float(res[0]) if np.isscalar(x) else res


def plot_interpolation(a, b, n, f, title=None):
    X_nodes = np.linspace(a, b, n+1)
    Y_nodes = f(X_nodes)
    X_plot = np.linspace(a, b, 2000)
    Y_true = f(X_plot)
    Y_poly = eval_newton(X_nodes, Y_nodes, X_plot)

    plt.figure(figsize=(9,5))
    plt.plot(X_plot, Y_true, label='f(x) vraie')
    plt.plot(X_plot, Y_poly, '--', label=f'Interpolant Newton (n={n})')
    plt.scatter(X_nodes, Y_nodes, c='red', s=20, label='nœuds')
    plt.legend()
    plt.grid(True)
    plt.title(title or f'Interpolation Newton n={n} sur [{a},{b}]')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()
    
def add_point_incremental(X_nodes, coeffs, x_new, y_new):
    """
    Ajoute un point sans recalculer toute la table.
    coeffs : coefficients Newton existants (a0,...,an)
    Retourne (X_new, coeffs_new).
    """
    X_nodes = list(X_nodes)
    coeffs = list(coeffs)

    # évaluer polynôme courant en x_new
    p_val = eval_newton(X_nodes, Y, x_new)

    # nouvelle base évaluée
    N_new = Base(X_nodes, len(X_nodes), x_new)

    # nouveau coefficient
    a_new = (y_new - p_val) / N_new

    # mise à jour
    X_nodes.append(x_new)
    coeffs.append(a_new)

    return np.array(X_nodes), np.array(coeffs)


if __name__ == '__main__':
    import math

    f1 = lambda x: np.sin(x)
    plot_interpolation(0.0, 2*math.pi, 10, f1, title='f = sin(x) — interpolation Newton (n=10)')

    f2 = lambda x: 1.0 / (1.0 + 10.0 * x**2)
    for n in (10, 20, 40):
        plot_interpolation(-5.0, 5.0, n, f2, title=f'Runge 1/(1+10 x^2) — n={n}')
