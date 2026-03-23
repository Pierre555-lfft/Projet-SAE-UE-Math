import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. MODÉLISATION DU PROFIL NACA
# ============================================================
def naca_4digits(x, m, p, t):
    """Génère l'extrados d'un profil NACA 4-digits."""
    # Épaisseur
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    # Ligne moyenne (Camber line)
    yc = np.where(x < p, 
                  (m / (p**2 + 1e-10)) * (2*p*x - x**2), 
                  (m / ((1-p)**2 + 1e-10)) * ((1-2*p) + 2*p*x - x**2))
    return yc + yt 

# ============================================================
# 2. ALGORITHME DE COX-DE BOOR (Correction des index)
# ============================================================
def deBoor(t, k, T, P):
    """Évalue la courbe B-spline au paramètre t."""
    # Gestion des bords pour éviter les index hors limites
    t = np.clip(t, T[k], T[-k-1] - 1e-10)
    
    # Trouver l'intervalle (knot span)
    j = np.searchsorted(T, t, side='right') - 1
    
    # Initialisation des points de travail
    d = [P[i].copy() for i in range(j - k, j + 1)]

    # Calcul récursif
    for r in range(1, k + 1):
        for i in range(k, r - 1, -1):
            alpha = (t - T[j - k + i]) / (T[j + i - r + 1] - T[j - k + i])
            d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
    return d[k]

# ============================================================
# 3. CONFIGURATION DE LA B-SPLINE
# ============================================================
k = 3 # Degré cubique
n_ctrl = 10 

# Correction du vecteur de nœuds (clamped spline)
knots = np.concatenate(([0]*(k+1), np.linspace(0, 1, n_ctrl-k+1)[1:-1], [1]*(k+1)))

# Points de contrôle initiaux
P_ctrl = np.zeros((n_ctrl, 2))
P_ctrl[:, 0] = np.linspace(0, 1, n_ctrl)
P_ctrl[:, 1] = 0.05 # Hauteur initiale

# Cible : NACA 8416
x_target = np.linspace(0, 1, 100)
#y_target = naca_4digits(x_target, 0.08, 0.4, 0.16)
y_target = naca_4digits(x_target, 1.2, 0.4, 0.16)

# ============================================================
# 4. OPTIMISATION PAR DESCENTE DE GRADIENT
# ============================================================
def calcul_cout(P):
    """Calcul de l'erreur quadratique entre la spline et la cible."""
    # On évalue la spline aux mêmes x que la cible
    # Note : u_param est ici simplifié à x_target pour la démo
    y_spline = np.array([deBoor(u, k, knots, P)[1] for u in x_target])
    return np.mean((y_spline - y_target)**2)

eta = 0.5 # Pas d'apprentissage augmenté pour la démo
iterations = 200
historique_cout = []

print("Optimisation de la voilure...")
for it in range(iterations):
    grad = np.zeros(n_ctrl)
    h = 1e-4
    
    current_cost = calcul_cout(P_ctrl)
    
    for i in range(1, n_ctrl - 1): # On ne touche pas aux extrémités
        P_temp = P_ctrl.copy()
        P_temp[i, 1] += h
        grad[i] = (calcul_cout(P_temp) - current_cost) / h
    
    P_ctrl[:, 1] -= eta * grad
    historique_cout.append(current_cost)

# ============================================================
# 5. AFFICHAGE DES RÉSULTATS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Graphique 1 : Géométrie
u_fine = np.linspace(0, 1, 200)
res = np.array([deBoor(u, k, knots, P_ctrl) for u in u_fine])
ax1.plot(x_target, y_target, 'r--', label="Cible (NACA 8416)", alpha=0.6)
ax1.plot(res[:, 0], res[:, 1], 'b', lw=2, label="Spline Optimisée")
ax1.scatter(P_ctrl[:, 0], P_ctrl[:, 1], c='black', s=20, label="Points de contrôle")
ax1.set_title("Géométrie du Profil")
ax1.legend(); ax1.grid(True); ax1.axis('equal')

# Graphique 2 : Convergence
ax2.plot(historique_cout, color='green')
ax2.set_yscale('log')
ax2.set_title("Convergence (Erreur Quadratique Moyenne)")
ax2.set_xlabel("Itérations")
ax2.grid(True)


plt.tight_layout()
plt.show()
