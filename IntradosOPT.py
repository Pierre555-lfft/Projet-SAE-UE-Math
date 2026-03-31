import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. MODÉLISATION DU PROFIL NACA
# ============================================================
def naca_4digits_intrados(x, m, p, t):
    """Génère l'intrados d'un profil NACA 4-digits."""
    # Épaisseur
    yt = 5 * t * (0.2969 * np.sqrt(x)- 0.1260 * x- 0.3516 * x**2+ 0.2843 * x**3- 0.1036 * x**4)

    # Ligne moyenne (camber line)
    yc = np.where(
        x < p,
        (m / (p**2 + 1e-10)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-10)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return yc - yt

# ============================================================
# 2. ALGORITHME DE DeBOOR
# ============================================================
def deBoor(t, k, T, P):
    """Évalue la courbe B-spline au paramètre t."""
    t = np.clip(t, T[k], T[-k-1] - 1e-10)
    j = np.searchsorted(T, t, side='right') - 1
    d = [P[i].copy() for i in range(j - k, j + 1)]

    for r in range(1, k + 1):
        for i in range(k, r - 1, -1):
            denom = T[j + i - r + 1] - T[j - k + i]
            alpha = 0.0 if abs(denom) < 1e-14 else (t - T[j - k + i]) / denom
            d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
    return d[k]

# ============================================================
# 3. CONFIGURATION DE LA B-SPLINE
# ============================================================
k = 3
n_ctrl = 10

noeuds = np.concatenate((
    [0] * (k + 1),
    np.linspace(0, 1, n_ctrl - k + 1)[1:-1],
    [1] * (k + 1)
))

P_ctrl = np.zeros((n_ctrl, 2))
P_ctrl[:, 0] = np.linspace(0, 1, n_ctrl)
P_ctrl[:, 1] = -0.05   # valeur initiale négative pour l'intrados

# Cible : intrados NACA 8416
x_target = np.linspace(0, 1, 100)
y_target = naca_4digits_intrados(x_target, 0.08, 0.4, 0.16)

# ============================================================
# 4. OPTIMISATION PAR DESCENTE DE GRADIENT
# ============================================================
def calcul_cout(P):
    y_spline = np.array([deBoor(u, k, noeuds, P)[1] for u in x_target])
    return np.mean((y_spline - y_target) ** 2)

eta = 0.5
iterations = 200
historique_cout = []

print("Optimisation de l'intrados...")
for it in range(iterations):
    grad = np.zeros(n_ctrl)
    h = 1e-4

    current_cost = calcul_cout(P_ctrl)

    for i in range(1, n_ctrl - 1):
        P_temp = P_ctrl.copy()
        P_temp[i, 1] += h
        grad[i] = (calcul_cout(P_temp) - current_cost) / h

    P_ctrl[:, 1] -= eta * grad
    historique_cout.append(current_cost)

# ============================================================
# 5. AFFICHAGE DES RÉSULTATS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

u_fine = np.linspace(0, 1, 200)
res = np.array([deBoor(u, k, noeuds, P_ctrl) for u in u_fine])

# Graphique 1 : Géométrie
ax1.plot(x_target, y_target, 'r--', label="Cible intrados (NACA 8416)", alpha=0.6)
ax1.plot(res[:, 0], res[:, 1], 'b', lw=2, label="Spline optimisée")
ax1.scatter(P_ctrl[:, 0], P_ctrl[:, 1], c='black', s=20, label="Points de contrôle")
ax1.set_title("Géométrie de l'intrados")
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Graphique 2 : Convergence
ax2.plot(historique_cout)
ax2.set_yscale('log')
ax2.set_title("Convergence (Erreur quadratique moyenne)")
ax2.set_xlabel("Itérations")
ax2.grid(True)

plt.tight_layout()
plt.show()