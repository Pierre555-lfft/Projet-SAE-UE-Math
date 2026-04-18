#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  # bibliothèque utilisée pour les calculs vectoriels et matriciels

# import des fonctions nécessaires pour construire le problème
# de moindres carrés et reconstruire la courbe B-spline
from bsplines import (
    construire_probleme_moindres_carres,
    reconstruire_points_controle,
    evaluer_courbe_bspline,
)


# fonction qui calcule la fonction coût du problème de moindres carrés
# f(x) = ||Ax - b||²
def fonction_cout(matrice_A, vecteur_b, vecteur_x):

    # calcule le résidu entre l'approximation Ax et les données b
    residu = matrice_A @ vecteur_x - vecteur_b

    # renvoie la norme au carré du résidu
    return float(residu @ residu)


# fonction qui calcule le gradient de la fonction coût
# ∇f(x) = 2 A^T (Ax - b)
def gradient_fonction_cout(matrice_A, vecteur_b, vecteur_x):

    # formule analytique du gradient pour un problème quadratique
    return 2.0 * matrice_A.T @ (matrice_A @ vecteur_x - vecteur_b)


# fonction qui calcule la Hessienne de la fonction coût
# H = 2 A^T A
def hessienne_fonction_cout(matrice_A):

    # dans ce problème quadratique, la Hessienne est constante
    return 2.0 * (matrice_A.T @ matrice_A)


# fonction qui implémente la descente de gradient
# x_{k+1} = x_k - pas * ∇f(x_k)
def descente_de_gradient(
    matrice_A,
    vecteur_b,
    point_initial,
    pas=1e-2,
    tolerance=1e-10,
    iterations_max=5000,
):
    # convertit le point initial en tableau numpy
    x = np.array(point_initial, dtype=float)

    # listes qui enregistrent l’évolution du coût et du gradient
    historique_cout = []
    historique_gradient = []

    # boucle principale de l’algorithme
    for k in range(iterations_max):

        # calcule le gradient au point courant
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)

        # calcule la norme du gradient (critère d’arrêt)
        norme_gradient = np.linalg.norm(gradient)

        # calcule la valeur courante de la fonction coût
        cout = fonction_cout(matrice_A, vecteur_b, x)

        # enregistre les valeurs dans les historiques
        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        # arrêt si le gradient est suffisamment petit
        if norme_gradient < tolerance:
            break

        # direction de descente = opposé du gradient
        direction_descente = -gradient

        # mise à jour du point courant avec un pas fixe
        x = x + pas * direction_descente

    # renvoie la solution et les historiques de convergence
    return x, historique_cout, historique_gradient, k + 1


# fonction qui implémente la méthode de Newton
# x_{k+1} = x_k - H^{-1} ∇f(x_k)
def methode_newton(
    matrice_A,
    vecteur_b,
    point_initial,
    tolerance=1e-10,
    iterations_max=100,
):
    # convertit le point initial en tableau numpy
    x = np.array(point_initial, dtype=float)

    # listes pour suivre la convergence
    historique_cout = []
    historique_gradient = []

    # la Hessienne est constante, donc on la calcule une seule fois
    hessienne = hessienne_fonction_cout(matrice_A)

    # boucle principale
    for k in range(iterations_max):

        # gradient au point courant
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)

        # norme du gradient
        norme_gradient = np.linalg.norm(gradient)

        # valeur de la fonction coût
        cout = fonction_cout(matrice_A, vecteur_b, x)

        # stockage pour affichage de convergence
        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        # arrêt si convergence
        if norme_gradient < tolerance:
            break

        # direction de Newton : résolution de H d = -grad
        direction_newton = -np.linalg.solve(hessienne, gradient)

        # mise à jour du point
        x = x + direction_newton

    # renvoie la solution et les historiques
    return x, historique_cout, historique_gradient, k + 1


# fonction qui implémente la méthode quasi-Newton BFGS
def methode_bfgs(
    matrice_A,
    vecteur_b,
    point_initial,
    pas_initial=1.0,
    tolerance=1e-10,
    iterations_max=300,
):
    # point de départ
    x = np.array(point_initial, dtype=float)

    # dimension du problème
    dimension = len(x)

    # initialisation de l’approximation de l’inverse de la Hessienne
    approximation_inverse_hessienne = np.eye(dimension)

    # listes de suivi de convergence
    historique_cout = []
    historique_gradient = []

    # boucle principale
    for k in range(iterations_max):

        # gradient au point courant
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)

        # norme du gradient
        norme_gradient = np.linalg.norm(gradient)

        # valeur du coût
        cout = fonction_cout(matrice_A, vecteur_b, x)

        # stockage des informations de convergence
        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        # arrêt si convergence
        if norme_gradient < tolerance:
            break

        # direction quasi-Newton obtenue avec l’approximation de H^{-1}
        direction = -approximation_inverse_hessienne @ gradient

        # initialisation du pas
        pas = pas_initial
        cout_courant = cout

        # backtracking : on réduit le pas tant que le coût ne diminue pas
        while pas > 1e-12:
            nouveau_x_test = x + pas * direction
            nouveau_cout_test = fonction_cout(matrice_A, vecteur_b, nouveau_x_test)

            # on garde le pas dès qu'il améliore la fonction coût
            if nouveau_cout_test < cout_courant:
                break

            # sinon on divise le pas par 2
            pas *= 0.5

        # nouveau point après mise à jour
        nouveau_x = x + pas * direction

        # gradient au nouveau point
        nouveau_gradient = gradient_fonction_cout(matrice_A, vecteur_b, nouveau_x)

        # variation des itérés
        delta = nouveau_x - x

        # variation des gradients
        y = nouveau_gradient - gradient

        # produit scalaire delta^T y
        produit = float(delta @ y)

        # mise à jour BFGS seulement si ce produit est positif
        # pour conserver une approximation définie positive
        if produit > 1e-14:

            # calcule S_k y_k
            Sy = approximation_inverse_hessienne @ y

            # premier terme de la mise à jour BFGS
            terme_1 = (
                1.0 + (y @ Sy) / produit
            ) * np.outer(delta, delta) / produit

            # second terme de la mise à jour BFGS
            terme_2 = (
                np.outer(delta, Sy) + np.outer(Sy, delta)
            ) / produit

            # nouvelle approximation de l’inverse de la Hessienne
            approximation_inverse_hessienne = (
                approximation_inverse_hessienne + terme_1 - terme_2
            )

        # on remplace l’ancien point par le nouveau
        x = nouveau_x

    # renvoie la solution et les historiques
    return x, historique_cout, historique_gradient, k + 1


# fonction générique qui ajuste un profil par une méthode
# itérative donnée (gradient, Newton ou BFGS)
def _ajuster_surface_iterative(points_cible, nb_points_controle, degre, solveur, nom_methode, **kwargs):

    # construit le problème de moindres carrés associé au profil
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(points_cible, nb_points_controle, degre)

    # point initial pour les coordonnées x :
    # interpolation linéaire entre le premier et le dernier point
    x0_interieur = np.linspace(point_initial[0], point_final[0], nb_points_controle)[1:-1]

    # point initial pour les coordonnées y
    y0_interieur = np.linspace(point_initial[1], point_final[1], nb_points_controle)[1:-1]

    # résolution pour les coordonnées x avec la méthode choisie
    solution_x, hist_cout_x, hist_grad_x, nb_iter_x = solveur(
        matrice_interieure, second_membre_x, x0_interieur, **kwargs
    )

    # résolution pour les coordonnées y avec la méthode choisie
    solution_y, hist_cout_y, hist_grad_y, nb_iter_y = solveur(
        matrice_interieure, second_membre_y, y0_interieur, **kwargs
    )

    # reconstruit tous les points de contrôle complets
    points_controle = reconstruire_points_controle(
        point_initial, point_final, solution_x, solution_y
    )

    # évalue la courbe B-spline finale
    courbe_approximee = evaluer_courbe_bspline(
        points_controle, degre, noeuds
    )

    # calcule la courbe approchée sur les mêmes points que les données d’origine
    approximation_sur_donnees = matrice_base_complete @ points_controle

    # calcule l’erreur quadratique moyenne
    erreur_moyenne = np.mean(
        np.sum((approximation_sur_donnees - points_cible) ** 2, axis=1)
    )

    # renvoie tous les résultats utiles
    return {
        "nom_methode": nom_methode,
        "points_controle": points_controle,
        "courbe": courbe_approximee,
        "erreur_moyenne": erreur_moyenne,
        "historique_cout_x": hist_cout_x,
        "historique_cout_y": hist_cout_y,
        "historique_gradient_x": hist_grad_x,
        "historique_gradient_y": hist_grad_y,
        "nombre_iterations_x": nb_iter_x,
        "nombre_iterations_y": nb_iter_y,
    }


# fonction qui ajuste un profil avec la descente de gradient
def ajuster_surface_par_gradient(points_cible, nb_points_controle, degre, pas=1e-2):

    # appelle la fonction générique avec le solveur gradient
    return _ajuster_surface_iterative(
        points_cible,
        nb_points_controle,
        degre,
        solveur=descente_de_gradient,
        nom_methode="Descente de gradient",
        pas=pas,
        tolerance=1e-12,
        iterations_max=10000,
    )


# fonction qui ajuste un profil avec la méthode de Newton
def ajuster_surface_par_newton(points_cible, nb_points_controle, degre):

    # appelle la fonction générique avec le solveur Newton
    return _ajuster_surface_iterative(
        points_cible,
        nb_points_controle,
        degre,
        solveur=methode_newton,
        nom_methode="Newton",
        tolerance=1e-12,
        iterations_max=50,
    )


# fonction qui ajuste un profil avec la méthode BFGS
def ajuster_surface_par_bfgs(points_cible, nb_points_controle, degre):

    # appelle la fonction générique avec le solveur BFGS
    return _ajuster_surface_iterative(
        points_cible,
        nb_points_controle,
        degre,
        solveur=methode_bfgs,
        nom_methode="BFGS",
        pas_initial=1.0,
        tolerance=1e-12,
        iterations_max=300,
    )