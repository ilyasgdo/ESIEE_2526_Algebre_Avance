import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


def base(X: np.ndarray, k: int, x: float) -> float:
    """Calcule la base de Lagrange L_k(x)"""
    P: float = 1.0
    for j in range(len(X)):
        if j != k:
            P *= (x - X[j]) / (X[k] - X[j])
    return P

def eval_lagrange(X: np.ndarray, Y: np.ndarray, x: float) -> float:
    """Évalue le polynôme d'interpolation de Lagrange en x"""
    S: float = 0.0
    for k in range(len(X)):
        S += Y[k] * base(X, k, x)
    return S
    

def compute_vandermonde(a: float, b: float, n: int, f
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les points d'interpolation, valeurs correspondantes, et coefficients du polynôme
    par la matrice de Vandermonde.
    """
    X_nodes = np.linspace(a, b, n+1)
    Y_nodes = f(X_nodes)
    V = np.vander(X_nodes, increasing=True)
    A = np.linalg.solve(V, Y_nodes)
    return X_nodes, Y_nodes, A


def evaluate_polynomial(A: np.ndarray, x: float) -> float:
    """
    Évalue le polynome P(x) défini par les coefficients A en utilisant l'algorithme de Horner.
    """
    result = 0.0
    for coef in reversed(A):
        result = result * x + coef
    return result

def evaluate_polynomial_vector(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Évalue le polynome P(x) pour un vecteur de points X.
    """
    return np.array([evaluate_polynomial(A, xi) for xi in X])


def max_error(f: Callable[[np.ndarray], np.ndarray], A: np.ndarray, X: np.ndarray
             ) -> Tuple[float, np.ndarray]:
    """
    Calcule l'écart maximum entre la fonction f et le polynome défini par A sur X.
    Retourne également le polynôme évalué sur X.
    """
    Y_true = f(X)
    Y_poly = evaluate_polynomial_vector(A, X)
    error = np.max(np.abs(Y_true - Y_poly))
    return error, Y_poly

def plot_interpolation(X_nodes: np.ndarray, Y_nodes: np.ndarray,
                       X_plot: np.ndarray, Y_poly: np.ndarray,
                       f) -> None:
    """
    print la fonction f, le polynome généré , et les points d interpolation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(X_plot, f(X_plot), label="f(x)", color="blue")
    plt.plot(X_plot, Y_poly, label="Polynome Vandermonde", color="red", linestyle="--")
    plt.scatter(X_nodes, Y_nodes, color="green", label="Points d interpolation")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolation polynomiale par Vandermonde")
    plt.grid(True)
    plt.show()
    
def plot_lagrange_interpolation(X_true: np.ndarray, Y_true: np.ndarray, X_lagrange: np.ndarray, Y_lagrange: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(X_true, Y_true, label="f(x)", color="blue")
    plt.plot(X_lagrange, Y_lagrange, label="Polynome Lagrange", color="red", linestyle="--")
    
def plot_lagrange_interpolation(X_true: np.ndarray, Y_true: np.ndarray, X_lagrange: np.ndarray, Y_lagrange: np.ndarray) -> None: 
    plt.figure(figsize=(10, 6)) 
    plt.plot(X_true, Y_true, label="f(x)", color="blue") 
    plt.plot(X_lagrange, Y_lagrange, label="Polynome Lagrange", color="red", linestyle="--") 
     
def test_lagrange(a: float, b: float, nb_pt: int, f, vectorise: bool) -> None: 
    """ 
    nb_pt est le paramètre cohérent avec compute_vandermonde : on prendra nb_pt+1 nœuds équi-espacés.
    Cette fonction trace la vraie fonction (grille dense) et le polynôme d'interpolation de Lagrange.
    """ 
    X_true = np.linspace(a, b, 1000) 
    if vectorise: 
        Y_true = f(X_true) 
    else: 
        Y_true = np.array([f(x) for x in X_true])   

    # --- CORRECTION IMPORTANTE : construire les nœuds d'interpolation, pas prendre X_true ---
    X_nodes = np.linspace(a, b, nb_pt + 1)    # nb_pt suivant la convention compute_vandermonde
    if vectorise:
        Y_nodes = f(X_nodes)
    else:
        Y_nodes = np.array([f(x) for x in X_nodes])

    # Évaluer le polynôme de Lagrange sur la grille dense X_true
    Y_lagrange = eval_lagrange(X_nodes, Y_nodes, X_true)

    plt.figure(figsize=(10, 6))
    plt.plot(X_true, Y_true, label="f(x)", color="red")
    plt.plot(X_true, Y_lagrange, label=f"Polynôme de Lagrange ({len(X_nodes)} nœuds)", color="blue")
    plt.scatter(X_nodes, Y_nodes, color="green", label="Points d'interpolation")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Interpolation polynomiale par Lagrange — {len(X_nodes)} nœuds — intervalle [{a}, {b}]")
    plt.grid(True)
    plt.show()
     
 
def main() -> None: 
    a: float = 0.0 
    b: float = 2 * math.pi 
    n: int = 10 
    n_runge = 20
    f = lambda x: np.sin(x) 
     
    X_nodes, Y_nodes, A = compute_vandermonde(a, b, n, f) 
     
    X_plot = np.linspace(a, b, 1000) 
    error, Y_poly = max_error(f, A, X_plot) 
     
    print(f"Écart maximum entre f(x) et le polynôme P(x) : {error:.6f}") 
     
    #plot_interpolation(X_nodes, Y_nodes, X_plot, Y_poly, f) 
     
    test_lagrange(a, b, n, f, True) 
    f2 = lambda x: 1/(1+x**2) 
     
    test_lagrange(-10, 10, n_runge, f2, False) 
     
if __name__ == "__main__": 
    main()