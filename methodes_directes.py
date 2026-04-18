#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  # bibliothèque utilisée pour le calcul matriciel

# import des fonctions utiles pour construire et évaluer la B-spline
from bsplines import (
    construire_probleme_moindres_carres,
    reconstruire_points_controle,
    evaluer_courbe_bspline,
)


# résolution du problème de moindres carrés par équation normale
def resoudre_equation_normale(matrice_A, vecteur_b):

    # résout le système :
    # (A^T A) x = A^T b
    # ce qui donne la solution des moindres carrés
    return np.linalg.solve(
        matrice_A.T @ matrice_A,
        matrice_A.T @ vecteur_b
    )


# résolution du problème de moindres carrés par factorisation QR
def resoudre_par_qr(matrice_A, vecteur_b):

    # décompose A sous la forme A = Q R
    # Q : matrice orthogonale
    # R : matrice triangulaire supérieure
    Q, R = np.linalg.qr(
        matrice_A,
        mode="reduced"
    )

    # résout ensuite le système R x = Q^T b
    return np.linalg.solve(
        R,
        Q.T @ vecteur_b
    )


# résolution du problème de moindres carrés par décomposition SVD
def resoudre_par_svd(matrice_A, vecteur_b):

    # décompose A sous la forme :
    # A = U Σ V^T
    U, valeurs_singulieres, Vt = np.linalg.svd(
        matrice_A,
        full_matrices=False
    )

    # construit l'inverse de la matrice diagonale Σ
    inverse_sigma = np.diag(
        1.0 / valeurs_singulieres
    )

    # calcule la solution des moindres carrés :
    # x = V Σ^{-1} U^T b
    return (
        Vt.T @ inverse_sigma @ U.T @ vecteur_b
    )


# fonction qui approxime un profil par B-spline avec la méthode
# de l’équation normale
def ajuster_surface_par_equation_normale(
        points_cible,
        nb_points_controle,
        degre):

    # construit le problème de moindres carrés AX ≈ b
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(
        points_cible,
        nb_points_controle,
        degre
    )

    # résout séparément pour les coordonnées x
    solution_x = resoudre_equation_normale(
        matrice_interieure,
        second_membre_x
    )

    # résout séparément pour les coordonnées y
    solution_y = resoudre_equation_normale(
        matrice_interieure,
        second_membre_y
    )

    # reconstruit tous les points de contrôle
    points_controle = reconstruire_points_controle(
        point_initial,
        point_final,
        solution_x,
        solution_y
    )

    # évalue la courbe B-spline finale
    courbe_approximee = evaluer_courbe_bspline(
        points_controle,
        degre,
        noeuds
    )

    # calcule les points reconstruits sur les données initiales
    approximation_sur_donnees = (
        matrice_base_complete @ points_controle
    )

    # calcule l’erreur quadratique moyenne entre courbe et points cibles
    erreur_moyenne = np.mean(
        np.sum(
            (approximation_sur_donnees - points_cible) ** 2,
            axis=1
        )
    )

    # renvoie les résultats dans un dictionnaire
    return {
        "nom_methode": "Équation normale",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }


# fonction qui approxime un profil par B-spline avec la méthode QR
def ajuster_surface_par_qr(
        points_cible,
        nb_points_controle,
        degre):

    # construit le problème de moindres carrés AX ≈ b
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(
        points_cible,
        nb_points_controle,
        degre
    )

    # résolution pour les coordonnées x
    solution_x = resoudre_par_qr(
        matrice_interieure,
        second_membre_x
    )

    # résolution pour les coordonnées y
    solution_y = resoudre_par_qr(
        matrice_interieure,
        second_membre_y
    )

    # reconstruction des points de contrôle complets
    points_controle = reconstruire_points_controle(
        point_initial,
        point_final,
        solution_x,
        solution_y
    )

    # évaluation de la courbe B-spline approchée
    courbe_approximee = evaluer_courbe_bspline(
        points_controle,
        degre,
        noeuds
    )

    # calcul de l’approximation sur les points d’origine
    approximation_sur_donnees = (
        matrice_base_complete @ points_controle
    )

    # calcul de l’erreur quadratique moyenne
    erreur_moyenne = np.mean(
        np.sum(
            (approximation_sur_donnees - points_cible) ** 2,
            axis=1
        )
    )

    # renvoie les résultats
    return {
        "nom_methode": "QR",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }


# fonction qui approxime un profil par B-spline avec la méthode SVD
def ajuster_surface_par_svd(
        points_cible,
        nb_points_controle,
        degre):

    # construit le problème de moindres carrés AX ≈ b
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(
        points_cible,
        nb_points_controle,
        degre
    )

    # résolution pour les coordonnées x
    solution_x = resoudre_par_svd(
        matrice_interieure,
        second_membre_x
    )

    # résolution pour les coordonnées y
    solution_y = resoudre_par_svd(
        matrice_interieure,
        second_membre_y
    )

    # reconstruction des points de contrôle
    points_controle = reconstruire_points_controle(
        point_initial,
        point_final,
        solution_x,
        solution_y
    )

    # évaluation de la courbe B-spline approchée
    courbe_approximee = evaluer_courbe_bspline(
        points_controle,
        degre,
        noeuds
    )

    # approximation de la courbe sur les points de départ
    approximation_sur_donnees = (
        matrice_base_complete @ points_controle
    )

    # calcul de l’erreur quadratique moyenne
    erreur_moyenne = np.mean(
        np.sum(
            (approximation_sur_donnees - points_cible) ** 2,
            axis=1
        )
    )

    # renvoie les résultats
    return {
        "nom_methode": "SVD",
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
    }