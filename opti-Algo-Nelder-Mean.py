#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============================================================
# HISTORIQUES
# ============================================================
CL_history = []
CD_history = []
J_history = []


# ============================================================
# 1. PROFIL NACA 4-DIGITS
# ============================================================
def extrados_naca_4_chiffres(x, m, p, t):
    yt = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return yc + yt


def intrados_naca_4_chiffres(x, m, p, t):
    yt = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return yc - yt


# ============================================================
# 2. FONCTIONS DE DÉFORMATION LISSES
# ============================================================
def bosse_gaussienne(x, center, width):
    return np.exp(-((x - center) / width) ** 2)


def deformer_profil(x, y_ext0, y_int0, params):
    """
    params = [a1, a2, b1, b2]
    a1, a2 : déformations extrados
    b1, b2 : déformations intrados
    """
    a1, a2, b1, b2 = params

    phi1 = bosse_gaussienne(x, center=0.30, width=0.18)
    phi2 = bosse_gaussienne(x, center=0.65, width=0.18)

    y_ext = y_ext0 + a1 * phi1 + a2 * phi2
    y_int = y_int0 + b1 * phi1 + b2 * phi2

    return y_ext, y_int


# ============================================================
# 3. CONSTRUCTION DU PROFIL COMPLET
# ============================================================
def construire_coordonnees_profil(x, y_ext, y_int):
    """
    Format XFOIL :
    bord de fuite -> extrados -> bord d'attaque -> intrados -> bord de fuite
    """
    x_full = np.concatenate([x[::-1], x[1:]])
    y_full = np.concatenate([y_ext[::-1], y_int[1:]])
    return x_full, y_full


# ============================================================
# 4. ÉCRITURE FICHIER .DAT
# ============================================================
def ecrire_fichier_profil(filename, x, y, name="Airfoil"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(name + "\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.8f} {yi:.8f}\n")


# ============================================================
# 5. PARSERS ROBUSTES
# ============================================================
def lire_fichier_polaire(polar_file):
    """
    Cherche une ligne numérique dans le fichier polaire.
    """
    if not os.path.exists(polar_file):
        return None, None, False

    try:
        with open(polar_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        numeric_candidates = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    alpha_val = float(parts[0])
                    cl_val = float(parts[1])
                    cd_val = float(parts[2])
                    numeric_candidates.append((alpha_val, cl_val, cd_val))
                except ValueError:
                    pass

        if len(numeric_candidates) == 0:
            return None, None, False

        alpha_val, cl_val, cd_val = numeric_candidates[-1]

        if np.isfinite(cl_val) and np.isfinite(cd_val):
            return cl_val, cd_val, True

    except Exception as e:
        print("Erreur lire_fichier_polaire :", e)

    return None, None, False


def extraire_cl_cd_stdout(stdout_text):
    """
    Cherche les dernières valeurs de CL et CD dans stdout,
    même si elles sont sur des lignes différentes.
    """
    try:
        num = r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"

        cl_matches = re.findall(rf"CL\s*=\s*{num}", stdout_text)
        cd_matches = re.findall(rf"CD\s*=\s*{num}", stdout_text)

        if len(cl_matches) == 0 or len(cd_matches) == 0:
            return None, None, False

        cl_val = float(cl_matches[-1])
        cd_val = float(cd_matches[-1])

        if np.isfinite(cl_val) and np.isfinite(cd_val):
            return cl_val, cd_val, True

    except Exception as e:
        print("Erreur extraire_cl_cd_stdout :", e)

    return None, None, False


# ============================================================
# 6. LANCEMENT XFOIL ROBUSTE
# ============================================================
def lancer_xfoil_robuste(airfoil_dat="airfoil.dat",
                         polar_file="polar.txt",
                         alpha=2.0,
                         Re=1e6,
                         n_iter=80,
                         debug=False):

    for fname in [polar_file, "xfoil_input.in"]:
        if os.path.exists(fname):
            os.remove(fname)

    commands = f"""PLOP
G F

LOAD {airfoil_dat}
NORM
PANE
OPER
VISC {Re}
ITER {n_iter}
PACC
{polar_file}

ALFA {alpha}
PACC
QUIT
"""

    with open("xfoil_input.in", "w", encoding="utf-8") as f:
        f.write(commands)

    try:
        with open("xfoil_input.in", "r", encoding="utf-8") as fin:
            result = subprocess.run(
                ["xfoil"],
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                env={**os.environ, "DISPLAY": ""}
            )
    except Exception as e:
        if debug:
            print("Erreur lancement XFOIL :", e)
        return None, None, False

    time.sleep(0.2)

    if debug:
        print("===== STDOUT =====")
        print(result.stdout)
        print("===== STDERR =====")
        print(result.stderr)

    # 1) priorité au fichier polar
    CL, CD, ok = lire_fichier_polaire(polar_file)
    if ok:
        return CL, CD, True

    # 2) sinon récupération via stdout
    CL, CD, ok = extraire_cl_cd_stdout(result.stdout)
    if ok:
        return CL, CD, True

    return None, None, False


# ============================================================
# 7. PÉNALITÉS GÉOMÉTRIQUES
# ============================================================
def penalite_geometrique(y_ext, y_int, params):
    penalty = 0.0

    thickness = y_ext - y_int

    # On ignore le premier et le dernier point
    # car au bord d'attaque / bord de fuite l'épaisseur peut être nulle
    thickness_inner = thickness[1:-1]

    min_thickness = np.min(thickness_inner)

    # extrados au-dessus de l'intrados dans la zone intérieure
    if min_thickness < 0:
        penalty += 500.0 + 500.0 * abs(min_thickness)

    # épaisseur minimale intérieure
    if min_thickness < 0.01:
        penalty += 100.0 * (0.01 - min_thickness) ** 2

    # bord de fuite pas trop ouvert
    te_gap = abs(y_ext[-1] - y_int[-1])
    if te_gap > 0.03:
        penalty += 50.0 * te_gap

    # amplitudes de déformation trop grandes
    penalty += 5.0 * np.sum(np.array(params) ** 2)

    # éviter des profils trop extrêmes
    if np.max(np.abs(y_ext)) > 0.30:
        penalty += 200.0
    if np.max(np.abs(y_int)) > 0.30:
        penalty += 200.0

    return float(penalty)


# ============================================================
# 8. FONCTION OBJECTIF
# ============================================================
def fonction_objectif(params, x, y_ext0, y_int0, alpha=2.0, Re=1e6, CL_target=0.5):
    global CL_history, CD_history, J_history

    try:
        y_ext, y_int = deformer_profil(x, y_ext0, y_int0, params)

        penalty_geom = penalite_geometrique(y_ext, y_int, params)

        x_full, y_full = construire_coordonnees_profil(x, y_ext, y_int)
        ecrire_fichier_profil("airfoil.dat", x_full, y_full, name="AirfoilOptim")

        CL, CD, ok = lancer_xfoil_robuste(
            airfoil_dat="airfoil.dat",
            polar_file="polar.txt",
            alpha=alpha,
            Re=Re,
            n_iter=80,
            debug=False
        )

        if (not ok) or (CL is None) or (CD is None):
            J = 1000.0 + penalty_geom
            J_history.append(J)
            return float(J)

        penalty_lift = 50.0 * (CL - CL_target) ** 2
        J = CD + penalty_lift + penalty_geom

        if not np.isfinite(J):
            J = 1000.0

        CL_history.append(CL)
        CD_history.append(CD)
        J_history.append(J)

        print(f"params = {params}, CL = {CL:.5f}, CD = {CD:.5f}, J = {J:.5f}")
        return float(J)

    except Exception as e:
        print("Erreur fonction_objectif :", e)
        J = 1000.0
        J_history.append(J)
        return float(J)


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Points cosinus
    # --------------------------------------------------------
    beta = np.linspace(0.0, np.pi, 160)
    x = 0.5 * (1.0 - np.cos(beta))

    # --------------------------------------------------------
    # Profil initial NACA
    # --------------------------------------------------------
    m = 0.08
    p = 0.4
    t = 0.16

    y_ext0 = extrados_naca_4_chiffres(x, m, p, t)
    y_int0 = intrados_naca_4_chiffres(x, m, p, t)

    # --------------------------------------------------------
    # Test initial
    # --------------------------------------------------------
    params0 = np.array([0.0, 0.0, 0.0, 0.0])

    print("Test initial...")
    J0 = fonction_objectif(params0, x, y_ext0, y_int0, alpha=2.0, Re=1e6, CL_target=0.5)
    print("J0 =", J0)

    # --------------------------------------------------------
    # Optimisation
    # --------------------------------------------------------
    print("\nDébut optimisation avec Nelder-Mead...\n")
    result = minimize(
        fonction_objectif,
        params0,
        args=(x, y_ext0, y_int0, 2.0, 1e6, 0.5),
        method="Nelder-Mead",
        options={
            "maxiter": 30,
            "xatol": 1e-3,
            "fatol": 1e-3,
            "disp": True
        }
    )

    print("\nOptimisation terminée")
    print("Succès :", result.success)
    print("Message :", result.message)
    print("Paramètres optimaux :", result.x)
    print("Coût final :", result.fun)

    # --------------------------------------------------------
    # Profil final
    # --------------------------------------------------------
    params_opt = result.x
    y_ext_opt, y_int_opt = deformer_profil(x, y_ext0, y_int0, params_opt)

    x_full_init, y_full_init = construire_coordonnees_profil(x, y_ext0, y_int0)
    x_full_opt, y_full_opt = construire_coordonnees_profil(x, y_ext_opt, y_int_opt)

    ecrire_fichier_profil("airfoil_init.dat", x_full_init, y_full_init, name="AirfoilInit")
    ecrire_fichier_profil("airfoil_optimise.dat", x_full_opt, y_full_opt, name="AirfoilOptimized")

    CL_final, CD_final, ok_final = lancer_xfoil_robuste(
        airfoil_dat="airfoil_optimise.dat",
        polar_file="polar_final.txt",
        alpha=2.0,
        Re=1e6,
        n_iter=80,
        debug=False
    )

    if ok_final:
        print(f"Résultat final : CL = {CL_final:.5f}, CD = {CD_final:.5f}")
    else:
        print("Impossible d'évaluer le profil final avec XFOIL.")

    # --------------------------------------------------------
    # Graphes
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(x_full_init, y_full_init, "--", label="Profil initial")
    plt.plot(x_full_opt, y_full_opt, label="Profil optimisé")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Profil initial et profil optimisé")
    plt.legend()

    plt.figure(figsize=(8, 4))
    plt.plot(J_history)
    plt.grid(True)
    plt.xlabel("Évaluation")
    plt.ylabel("J")
    plt.title("Évolution du coût")

    plt.figure(figsize=(8, 4))
    if len(CL_history) > 0:
        plt.plot(CL_history, label="CL")
    if len(CD_history) > 0:
        plt.plot(CD_history, label="CD")
    plt.grid(True)
    plt.xlabel("Évaluation")
    plt.ylabel("Valeur")
    plt.title("Évolution de CL et CD")
    plt.legend()

    if len(CL_history) > 0 and len(CD_history) > 0:
        finesse = np.array(CL_history) / np.maximum(np.array(CD_history), 1e-12)
        plt.figure(figsize=(8, 4))
        plt.plot(finesse)
        plt.grid(True)
        plt.xlabel("Évaluation")
        plt.ylabel("CL / CD")
        plt.title("Évolution de la finesse")

    plt.tight_layout()
    plt.show()