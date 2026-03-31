#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:34:27 2026

@author: cytech
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PROFIL NACA 4-DIGITS
# ============================================================
def naca_4digits_extrados(x, m, p, t):
    """Génère l'extrados d'un profil NACA 4-digits."""
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.where(
        x < p,
        (m / (p**2 + 1e-10)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-10)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return yc + yt


def naca_4digits_intrados(x, m, p, t):
    """Génère l'intrados d'un profil NACA 4-digits."""
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.where(
        x < p,
        (m / (p**2 + 1e-10)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-10)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return yc - yt


# ============================================================
# 2. ALGORITHME DE COX-DE BOOR
# ============================================================
def deBoor(t, k, T, P):
    """Évalue la courbe B-spline au paramètre t."""
    t = np.clip(t, T[k], T[-k - 1] - 1e-10)
    j = np.searchsorted(T, t, side='right') - 1
    d = [P[i].copy() for i in range(j - k, j + 1)]

    for r in range(1, k + 1):
        for i in range(k, r - 1, -1):
            denom = T[j + i - r + 1] - T[j - k + i]
            alpha = 0.0 if abs(denom) < 1e-14 else (t - T[j - k + i]) / denom
            d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]

    return d[k]


# ============================================================
# 3. FONCTION GÉNÉRIQUE D'OPTIMISATION
# ============================================================
def optimise_surface(y_target, x_target, y_init, k=3, n_ctrl=10, eta=0.5, iterations=200):
    """Optimise une B-spline pour approximer une cible."""
    noeuds = np.concatenate((
        [0] * (k + 1),
        np.linspace(0, 1, n_ctrl - k + 1)[1:-1],
        [1] * (k + 1)
    ))

    P_ctrl = np.zeros((n_ctrl, 2))
    P_ctrl[:, 0] = np.linspace(0, 1, n_ctrl)
    P_ctrl[:, 1] = y_init

    def calcul_cout(P):
        y_spline = np.array([deBoor(u, k, noeuds, P)[1] for u in x_target])
        return np.mean((y_spline - y_target) ** 2)

    historique_cout = []

    for _ in range(iterations):
        grad = np.zeros(n_ctrl)
        h = 1e-4
        current_cost = calcul_cout(P_ctrl)

        for i in range(1, n_ctrl - 1):  # on ne touche pas aux extrémités
            P_temp = P_ctrl.copy()
            P_temp[i, 1] += h
            grad[i] = (calcul_cout(P_temp) - current_cost) / h

        P_ctrl[:, 1] -= eta * grad
        historique_cout.append(current_cost)

    return P_ctrl, noeuds, historique_cout


# ============================================================
# 4. PARAMÈTRES DU PROFIL
# ============================================================
m = 0.08
p = 0.4
t = 0.16

x_target = np.linspace(0, 1, 100)

# Cibles NACA
y_target_ext = naca_4digits_extrados(x_target, m, p, t)
y_target_int = naca_4digits_intrados(x_target, m, p, t)

# Optimisation extrados et intrados
P_ctrl_ext, noeuds_ext, hist_ext = optimise_surface(
    y_target_ext, x_target, y_init=0.05, k=3, n_ctrl=10, eta=0.5, iterations=200
)

P_ctrl_int, noeuds_int, hist_int = optimise_surface(
    y_target_int, x_target, y_init=-0.05, k=3, n_ctrl=10, eta=0.5, iterations=200
)

# Évaluation fine
u_fine = np.linspace(0, 1, 300)
res_ext = np.array([deBoor(u, 3, noeuds_ext, P_ctrl_ext) for u in u_fine])
res_int = np.array([deBoor(u, 3, noeuds_int, P_ctrl_int) for u in u_fine])

# ============================================================
# 5. AFFICHAGE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Graphe 1 : profil complet ---
ax1 = axes[0]
ax1.plot(x_target, y_target_ext, 'r--', label="Cible extrados", alpha=0.7)
ax1.plot(x_target, y_target_int, 'm--', label="Cible intrados", alpha=0.7)

ax1.plot(res_ext[:, 0], res_ext[:, 1], 'b', lw=2, label="Spline extrados optimisée")
ax1.plot(res_int[:, 0], res_int[:, 1], 'g', lw=2, label="Spline intrados optimisée")

ax1.scatter(P_ctrl_ext[:, 0], P_ctrl_ext[:, 1], c='black', s=20, label="Pts contrôle extrados")
ax1.scatter(P_ctrl_int[:, 0], P_ctrl_int[:, 1], c='orange', s=20, label="Pts contrôle intrados")

# fermer visuellement le profil
ax1.plot(
    [res_ext[0, 0], res_int[0, 0]],
    [res_ext[0, 1], res_int[0, 1]],
    'k-', lw=1
)
ax1.plot(
    [res_ext[-1, 0], res_int[-1, 0]],
    [res_ext[-1, 1], res_int[-1, 1]],
    'k-', lw=1
)

ax1.set_title("Profil complet : extrados + intrados")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

# --- Graphe 2 : convergence ---
ax2 = axes[1]
ax2.plot(hist_ext, label="Erreur extrados")
ax2.plot(hist_int, label="Erreur intrados")
ax2.set_yscale('log')
ax2.set_title("Convergence des erreurs")
ax2.set_xlabel("Itérations")
ax2.set_ylabel("Erreur quadratique moyenne")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()