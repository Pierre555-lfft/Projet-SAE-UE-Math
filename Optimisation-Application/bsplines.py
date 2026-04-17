

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # importes la bibliothèque NumPy sous le nom np


# fonction qui calcule une fonction de base de B-spline
def base_bspline(indice, degre, noeuds, u):
    if degre == 0: # cas de la fonction constante
        if (noeuds[indice] <= u < noeuds[indice + 1]) or (
            abs(u - 1.0) < 1e-14 and abs(noeuds[indice + 1] - 1.0) < 1e-14
        ):
            return 1.0
        return 0.0

    terme_gauche = 0.0
    terme_droite = 0.0

    denominateur_1 = noeuds[indice + degre] - noeuds[indice]
    denominateur_2 = noeuds[indice + degre + 1] - noeuds[indice + 1]

    if abs(denominateur_1) > 1e-14:
        terme_gauche = ((u - noeuds[indice]) / denominateur_1) * base_bspline(
            indice, degre - 1, noeuds, u
        )

    if abs(denominateur_2) > 1e-14:
        terme_droite = ((noeuds[indice + degre + 1] - u) / denominateur_2) * base_bspline(
            indice + 1, degre - 1, noeuds, u
        )

    return terme_gauche + terme_droite

# fonction qui permet de construire la matrice A des fonctions de base
def construire_matrice_base(valeurs_u, nb_points_controle, degre, noeuds):
    matrice = np.zeros((len(valeurs_u), nb_points_controle), dtype=float)

    for i, u in enumerate(valeurs_u):
        for j in range(nb_points_controle):
            matrice[i, j] = base_bspline(j, degre, noeuds, float(u))

    return matrice


# fonction qui permet de construire le vecteur de noeuds de la B-spline
def construire_noeuds_ouverts_uniformes(nb_points_controle, degre):
    nb_noeuds = nb_points_controle + degre + 1
    noeuds = np.zeros(nb_noeuds)

    noeuds[:degre + 1] = 0.0
    noeuds[-(degre + 1):] = 1.0

    nb_noeuds_internes = nb_noeuds - 2 * (degre + 1)
    if nb_noeuds_internes > 0:
        noeuds_internes = np.linspace(0.0, 1.0, nb_noeuds_internes + 2)[1:-1]
        noeuds[degre + 1:-(degre + 1)] = noeuds_internes

    return noeuds


# fonction qui permet d'approximer un nuage de point par une B-spline
def construire_probleme_moindres_carres(points_cible, nb_points_controle, degre):
    nb_points = len(points_cible)
    valeurs_u = np.linspace(0.0, 1.0, nb_points)

    noeuds = construire_noeuds_ouverts_uniformes(nb_points_controle, degre)
    matrice_base = construire_matrice_base(valeurs_u, nb_points_controle, degre, noeuds)

    c0 = matrice_base[:, 0]
    cn = matrice_base[:, -1]
    matrice_interieure = matrice_base[:, 1:-1]

    point_initial = points_cible[0].copy()
    point_final = points_cible[-1].copy()

    second_membre_x = points_cible[:, 0] - point_initial[0] * c0 - point_final[0] * cn
    second_membre_y = points_cible[:, 1] - point_initial[1] * c0 - point_final[1] * cn

    return (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base,
    )


# fonction qui permet de reconstruire les points de contrôle après l'optimisation 
def reconstruire_points_controle(point_initial, point_final, x_interieur, y_interieur):
    nb_points_controle = len(x_interieur) + 2
    points_controle = np.zeros((nb_points_controle, 2), dtype=float)

    points_controle[0] = point_initial
    points_controle[-1] = point_final
    points_controle[1:-1, 0] = x_interieur
    points_controle[1:-1, 1] = y_interieur

    return points_controle


# fonction qui permet d'évaluer la courbe B-spline finale pour pouvoir la tracer 
def evaluer_courbe_bspline(points_controle, degre, noeuds, nb_points=400):
    valeurs_u = np.linspace(0.0, 1.0, nb_points)
    matrice = construire_matrice_base(valeurs_u, len(points_controle), degre, noeuds)
    return matrice @ points_controle