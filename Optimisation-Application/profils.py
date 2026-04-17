

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # importes la bibliothèque NumPy sous le nom np

# fonction qui calcule l'extrados (dessus du profil)
def naca_4_chiffres_extrados(x, m, p, t):
    # formule qui calcule la distribution d'épaisseur du profil NACA
    epaisseur = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12)) # remplace les petites valeur par 1e-12 et évite des instabilités numérique
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    ligne_cambre = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2), # + 1e-12 est là pour évite que p ou 1-p devienne très petit
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2), # + 1e-12 est là pour évite que p ou 1-p devienne très petit
    )

    return ligne_cambre + epaisseur

# fonction qui calcule l'intrados (dessous du profil)
def naca_4_chiffres_intrados(x, m, p, t):
    # formule qui calcule la distribution d'épaisseur du profil NACA
    epaisseur = 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12)) # remplace les petites valeur par 1e-12 et évite des instabilités numérique
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    ligne_cambre = np.where(
        x < p,
        (m / (p**2 + 1e-12)) * (2 * p * x - x**2), # + 1e-12 est là pour évite que p ou 1-p devienne très petit
        (m / ((1 - p)**2 + 1e-12)) * ((1 - 2 * p) + 2 * p * x - x**2), # + 1e-12 est là pour évite que p ou 1-p devienne très petit
    )

    return ligne_cambre - epaisseur

# fonction qui sert à générer totalement le profil NACA
def generer_profil_naca(m, p, t, nb_points=120): # 120 : nombre de point utiliser pour la discrétisation du profil
    beta = np.linspace(0.0, np.pi, nb_points) # construire un tableau beta allant de 0 à Pi, avec nb_points valeurs
    x = 0.5 * (1.0 - np.cos(beta)) # transforme beta en abscisses x sur [0,1]

    y_extrados = naca_4_chiffres_extrados(x, m, p, t)
    y_intrados = naca_4_chiffres_intrados(x, m, p, t)

    points_extrados = np.column_stack([x, y_extrados]) # on regroupe ici x et y_extrados en ligne chaque ligne représente un point du profil 
    points_intrados = np.column_stack([x, y_intrados]) # on regroupe ici x et y_intrados en ligne chaque ligne représente un point du profil

    return points_extrados, points_intrados # renvoie les points utilisable par la suite 