

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from bsplines import (
    construire_probleme_moindres_carres,
    reconstruire_points_controle,
    evaluer_courbe_bspline,
)

# fonction de résulotion par méthode de l'équation normal 
def resoudre_equation_normale(matrice_A, vecteur_b):
    return np.linalg.solve(matrice_A.T @ matrice_A, matrice_A.T @ vecteur_b)


# fonction de résolution par méthode de factorisation QR
def resoudre_par_qr(matrice_A, vecteur_b):
    Q, R = np.linalg.qr(matrice_A, mode="reduced")
    return np.linalg.solve(R, Q.T @ vecteur_b)


# focntion de résolution par méthode de décomposition en valeur singulière 
def resoudre_par_svd(matrice_A, vecteur_b):
    U, valeurs_singulieres, Vt = np.linalg.svd(matrice_A, full_matrices=False)
    inverse_sigma = np.diag(1.0 / valeurs_singulieres)
    return Vt.T @ inverse_sigma @ U.T @ vecteur_b


# fonction qui Approxime un ensemble de points (profil) par une courbe B-spline en résolvant un problème de moindres carrés avec la méthode de l'équation normal.
def ajuster_surface_par_equation_normale(points_cible, nb_points_controle, degre):
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(points_cible, nb_points_controle, degre)

    solution_x = resoudre_equation_normale(matrice_interieure, second_membre_x)
    solution_y = resoudre_equation_normale(matrice_interieure, second_membre_y)

    points_controle = reconstruire_points_controle(point_initial, point_final, solution_x, solution_y)
    courbe_approximee = evaluer_courbe_bspline(points_controle, degre, noeuds)

    approximation_sur_donnees = matrice_base_complete @ points_controle
    erreur_moyenne = np.mean(np.sum((approximation_sur_donnees - points_cible) ** 2, axis=1))

    return {
        "nom_methode": "Équation normale",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }

# fonction qui Approxime un ensemble de points (profil) par une courbe B-spline en résolvant un problème de moindres carrés avec la méthode QR.
def ajuster_surface_par_qr(points_cible, nb_points_controle, degre):
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(points_cible, nb_points_controle, degre)

    solution_x = resoudre_par_qr(matrice_interieure, second_membre_x)
    solution_y = resoudre_par_qr(matrice_interieure, second_membre_y)

    points_controle = reconstruire_points_controle(point_initial, point_final, solution_x, solution_y)
    courbe_approximee = evaluer_courbe_bspline(points_controle, degre, noeuds)

    approximation_sur_donnees = matrice_base_complete @ points_controle
    erreur_moyenne = np.mean(np.sum((approximation_sur_donnees - points_cible) ** 2, axis=1))

    return {
        "nom_methode": "QR",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }

# fonction qui Approxime un ensemble de points (profil) par une courbe B-spline en résolvant un problème de moindres carrés avec la méthode SVD.
def ajuster_surface_par_svd(points_cible, nb_points_controle, degre):
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(points_cible, nb_points_controle, degre)

    solution_x = resoudre_par_svd(matrice_interieure, second_membre_x)
    solution_y = resoudre_par_svd(matrice_interieure, second_membre_y)

    points_controle = reconstruire_points_controle(point_initial, point_final, solution_x, solution_y)
    courbe_approximee = evaluer_courbe_bspline(points_controle, degre, noeuds)

    approximation_sur_donnees = matrice_base_complete @ points_controle
    erreur_moyenne = np.mean(np.sum((approximation_sur_donnees - points_cible) ** 2, axis=1))

    return {
        "nom_methode": "SVD",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }