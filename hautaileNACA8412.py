import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.linalg import lstsq

def naca_4digits(x, m, p, t):
    """Génère les coordonnées exactes d'un profil NACA 4 chiffres."""
    # Distribution d'épaisseur (Source p. 16/23)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
                  0.2843*x**3 - 0.1036*x**4)
    
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    
    # Ligne de cambrure et pente (Source p. 23)
    mask = x <= p
    if p > 0:
        yc[mask] = (m / p**2) * (2*p*x[mask] - x[mask]**2)
        dyc[mask] = (2*m / p**2) * (p - x[mask])
        yc[~mask] = (m / (1-p)**2) * ((1-2*p) + 2*p*x[~mask] - x[~mask]**2)
        dyc[~mask] = (2*m / (1-p)**2) * (p - x[~mask])
    
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

# Paramétrage
N_data = 200
n_ctrl = 20
k = 3
u = np.linspace(0, 1, N_data)
xu_naca, yu_naca, xl_naca, yl_naca = naca_4digits(u, 0.08, 0.4, 0.12)

# Vecteur de nœuds vissé (Multiplicité k+1 aux bords)
knots = np.zeros(n_ctrl + k + 1)
knots[k:n_ctrl+1] = np.linspace(0, 1, n_ctrl - k + 1)
knots[n_ctrl+1:] = 1.0

# Construction de la matrice de base A
A = np.zeros((N_data, n_ctrl))
for j in range(n_ctrl):
    coeffs = np.zeros(n_ctrl)
    coeffs[j] = 1.0
    A[:, j] = BSpline(knots, coeffs, k)(u)

def fit_profile(x_data, y_data):
    A_int = A[:, 1:-1]
    # RHS avec soustraction des contributions des extrémités fixées
    rhs_x = x_data - A[:, 0]*x_data[0] - A[:, -1]*x_data[-1]
    rhs_y = y_data - A[:, 0]*y_data[0] - A[:, -1]*y_data[-1]
    
    xi, _, _, _ = lstsq(A_int, rhs_x)
    yi, _, _, _ = lstsq(A_int, rhs_y)
    
    X_ctrl = np.concatenate(([x_data[0]], xi, [x_data[-1]]))
    Y_ctrl = np.concatenate(([y_data[0]], yi, [y_data[-1]]))
    return X_ctrl, Y_ctrl

# Fitting de l'extrados
Xc, Yc = fit_profile(xu_naca, yu_naca)
sp = BSpline(knots, np.column_stack((Xc, Yc)), k)
u_fine = np.linspace(0, 1, 500)
fitted_curve = sp(u_fine)

# Visualisation
plt.figure(figsize=(10, 4))
plt.plot(xu_naca, yu_naca, 'r--', label='NACA 8412 Théorique')
plt.plot(fitted_curve[:, 0], fitted_curve[:, 1], 'b-', label='Approximation B-spline')
plt.scatter(Xc, Yc, color='black', s=20, label='Points de Contrôle')
plt.axis('equal')
plt.legend()
plt.title("Modélisation B-spline de l'extrados (NACA 8412)")
plt.show()
