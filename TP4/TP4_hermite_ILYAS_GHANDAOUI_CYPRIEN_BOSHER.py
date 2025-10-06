import numpy as np
import matplotlib.pyplot as plt

def phi1(x):
    if x<0 or x>1 :
        return 0

    return (2*x+1)*(x-1)**2
def phi2(x):
    if x<0 or x>1 :
        return 0

    return x**2*(-2*x+3)

def phi3(x):
    if x<0 or x>1 :
        return 0

    return x*(x-1)**2

def phi4(x):
    if x<0 or x>1 :
        return 0

    return x**2*(x-1)





def foncHermite(X, Y, V, x):
   
    
    
    if len(X) == 0:
        return 0
    
    S = 0  
    
    for i in range(len(X) - 1):
        if X[i] <= x <= X[i + 1]:
            delta = X[i + 1] - X[i]  
            t = (x - X[i]) / delta    
            
            S+= (Y[i] * phi1(t) + 
                   Y[i + 1] * phi2(t) + 
                   delta * V[i] * phi3(t) + 
                   delta * V[i + 1] * phi4(t))
            
            break  
    
    return S

def plot_hermite_interpolation(X, Y, V):
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    V = np.asarray(V)
    
    Xmin, Xmax = X.min(), X.max()
    x_plot = np.linspace(Xmin, Xmax, 1000)  
    y_appr = [foncHermite(X, Y, V, xi) for xi in x_plot]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_appr, 'b-' ,label="interpolation hermite")
    plt.scatter(X, Y, color='red', s=50, zorder=5, label='pt interpolation')
    
    for i in range(len(X)):

        dx = 0.2 * (Xmax - Xmin) / len(X) 
        x_tangent = [X[i] - dx, X[i] + dx]
        y_tangent = [Y[i] - V[i] * dx, Y[i] + V[i] * dx]


        plt.plot(x_tangent, y_tangent, 'g--', alpha=0.7, linewidth=1)
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title("Interpolation hermite")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    import math

   

    X=[-5,-2,0,3,6]
    Y=[-4,-1,1,1,-1]
    V=[3,0,3,-2,0]
    plot_hermite_interpolation(X, Y, V)


    X=[-5,-2,3, 7]
    Y=[0,-3,4,3]
    V=[0,1,1.5,1.8]
    plot_hermite_interpolation(X, Y, V)


