import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# -------------------------------
# Calcul Vandermonde
# -------------------------------
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

# -------------------------------
# Évaluation du polynome avec Horner
# -------------------------------
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

# -------------------------------
# Calcul de l'écart maximum
# -------------------------------
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

# -------------------------------
# Affichage graphique
# -------------------------------
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

# -------------------------------
# Main
# -------------------------------
def main() -> None:
    a: float = 0.0
    b: float = 2 * math.pi
    n: int = 3
    f = lambda x: np.sin(x)
    
    X_nodes, Y_nodes, A = compute_vandermonde(a, b, n, f)
    
    X_plot = np.linspace(a, b, 1000)
    error, Y_poly = max_error(f, A, X_plot)
    
    print(f"Écart maximum entre f(x) et le polynôme P(x) : {error:.6f}")
    
    plot_interpolation(X_nodes, Y_nodes, X_plot, Y_poly, f)

if __name__ == "__main__":
    main()
