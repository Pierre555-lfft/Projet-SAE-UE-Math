#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HISTORIQUES
# ============================================================
historique_CL = []
historique_CD = []
historique_cout = []
historique_norme_gradient = []
historique_meilleur_cout = []


# ============================================================
# 1. PROFIL NACA 4 CHIFFRES
# ============================================================
def naca_4_chiffres_extrados(x, m, p, t):
    epaisseur = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    ligne_cambre = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return ligne_cambre + epaisseur


def naca_4_chiffres_intrados(x, m, p, t):
    epaisseur = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    ligne_cambre = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2),
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2)
    )

    return ligne_cambre - epaisseur


# ============================================================
# 2. DÉFORMATION DU PROFIL
# ============================================================
def bosse_gaussienne(x, centre, largeur):
    return np.exp(-((x - centre) / largeur) ** 2)


def deformer_profil(x, y_ext0, y_int0, parametres):
    a1, a2, b1, b2 = parametres

    phi1 = bosse_gaussienne(x, centre=0.30, largeur=0.18)
    phi2 = bosse_gaussienne(x, centre=0.65, largeur=0.18)

    y_ext = y_ext0 + a1 * phi1 + a2 * phi2
    y_int = y_int0 + b1 * phi1 + b2 * phi2

    return y_ext, y_int


# ============================================================
# 3. CONSTRUCTION DU PROFIL COMPLET
# ============================================================
def construire_coordonnees_profil(x, y_ext, y_int):
    x_total = np.concatenate([x[::-1], x[1:]])
    y_total = np.concatenate([y_ext[::-1], y_int[1:]])
    return x_total, y_total


# ============================================================
# 4. ÉCRITURE FICHIER
# ============================================================
def ecrire_fichier_profil(nom_fichier, x, y, nom="Profil"):
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(nom + "\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.8f} {yi:.8f}\n")


# ============================================================
# 5. EXTRACTION DES RÉSULTATS XFOIL
# ============================================================
def lire_polaire(fichier):
    if not os.path.exists(fichier):
        return None, None, False

    try:
        with open(fichier, "r") as f:
            lignes = f.readlines()

        valeurs = []
        for ligne in lignes:
            parties = ligne.split()
            if len(parties) >= 3:
                try:
                    cl = float(parties[1])
                    cd = float(parties[2])
                    valeurs.append((cl, cd))
                except:
                    pass

        if len(valeurs) == 0:
            return None, None, False

        return valeurs[-1][0], valeurs[-1][1], True

    except:
        return None, None, False


# ============================================================
# 6. LANCEMENT XFOIL
# ============================================================
def lancer_xfoil(fichier_profil="profil.dat", fichier_polaire="polaire.txt",
                 alpha=2.0, Re=1e6, iterations=80):

    commandes = f"""PLOP
G F
LOAD {fichier_profil}
NORM
PANE
OPER
VISC {Re}
ITER {iterations}
PACC
{fichier_polaire}

ALFA {alpha}
PACC
QUIT
"""

    with open("input_xfoil.in", "w") as f:
        f.write(commandes)

    subprocess.run(
        ["xfoil"],
        stdin=open("input_xfoil.in"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(0.2)

    return lire_polaire(fichier_polaire)


# ============================================================
# 7. PÉNALITÉS
# ============================================================
def penalite_geometrique(y_ext, y_int, parametres):
    penalite = 0.0

    epaisseur = y_ext - y_int
    ep_min = np.min(epaisseur[1:-1])

    if ep_min < 0:
        penalite += 500

    if ep_min < 0.01:
        penalite += 100 * (0.01 - ep_min) ** 2

    penalite += 5 * np.sum(parametres**2)

    return penalite


# ============================================================
# 8. FONCTION COÛT
# ============================================================
def calculer_cout(parametres, x, y_ext0, y_int0,
                  alpha=2.0, Re=1e6, CL_cible=0.5,
                  stocker=False):

    y_ext, y_int = deformer_profil(x, y_ext0, y_int0, parametres)

    penalite = penalite_geometrique(y_ext, y_int, parametres)

    x_total, y_total = construire_coordonnees_profil(x, y_ext, y_int)
    ecrire_fichier_profil("profil.dat", x_total, y_total)

    CL, CD, ok = lancer_xfoil()

    if not ok:
        return 1000 + penalite, None, None, False

    cout = CD + 50 * (CL - CL_cible)**2 + penalite

    if stocker:
        historique_CL.append(CL)
        historique_CD.append(CD)
        historique_cout.append(cout)

    return cout, CL, CD, True


# ============================================================
# 9. GRADIENT NUMÉRIQUE
# ============================================================
def calculer_gradient(parametres, x, y_ext0, y_int0,
                      alpha, Re, CL_cible, h=1e-4):

    grad = np.zeros(len(parametres))
    J0, _, _, _ = calculer_cout(parametres, x, y_ext0, y_int0)

    for i in range(len(parametres)):
        p = parametres.copy()
        p[i] += h

        Ji, _, _, _ = calculer_cout(p, x, y_ext0, y_int0)

        grad[i] = (Ji - J0) / h

    return grad, J0


# ============================================================
# 10. DESCENTE DE GRADIENT
# ============================================================
def descente_gradient(param_init, x, y_ext0, y_int0,
                      alpha=2.0, Re=1e6, CL_cible=0.5):

    params = param_init.copy()

    for i in range(10):
        grad, J = calculer_gradient(params, x, y_ext0, y_int0)

        norme = np.linalg.norm(grad)
        historique_norme_gradient.append(norme)

        print(f"it {i}, J={J:.4f}, ||grad||={norme:.4f}")

        params = params - 0.01 * grad

    return params


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    beta = np.linspace(0, np.pi, 160)
    x = 0.5 * (1 - np.cos(beta))

    y_ext0 = naca_4_chiffres_extrados(x, 0.08, 0.4, 0.16)
    y_int0 = naca_4_chiffres_intrados(x, 0.08, 0.4, 0.16)

    params0 = np.zeros(4)

    params_opt = descente_gradient(params0, x, y_ext0, y_int0)

    print("Paramètres optimaux :", params_opt)