

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from bsplines import (
    construire_probleme_moindres_carres,
    reconstruire_points_controle,
    evaluer_courbe_bspline,
)

# fonction qui calcule le coût du problème de moindres carrés 
def fonction_cout(matrice_A, vecteur_b, vecteur_x):
    residu = matrice_A @ vecteur_x - vecteur_b
    return float(residu @ residu)


# fonction qui calcul le gradient de la fonction coût 
def gradient_fonction_cout(matrice_A, vecteur_b, vecteur_x):
    return 2.0 * matrice_A.T @ (matrice_A @ vecteur_x - vecteur_b)


# focntion qui permet de calculer la fonction hermissienne de la fonction coût 
def hessienne_fonction_cout(matrice_A):
    return 2.0 * (matrice_A.T @ matrice_A)

# fonction qui implémente la méthode de descente de gradient pour minimiser la fonction coût 
def descente_de_gradient(
    matrice_A,
    vecteur_b,
    point_initial,
    pas=1e-2,
    tolerance=1e-10,
    iterations_max=5000,
):
    x = np.array(point_initial, dtype=float)

    historique_cout = []
    historique_gradient = []

    for k in range(iterations_max):
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)
        norme_gradient = np.linalg.norm(gradient)
        cout = fonction_cout(matrice_A, vecteur_b, x)

        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        if norme_gradient < tolerance:
            break

        direction_descente = -gradient
        x = x + pas * direction_descente

    return x, historique_cout, historique_gradient, k + 1


# fonction qui implémente la méthode de newton pour minimiser la fonction coût
def methode_newton(
    matrice_A,
    vecteur_b,
    point_initial,
    tolerance=1e-10,
    iterations_max=100,
):
    x = np.array(point_initial, dtype=float)

    historique_cout = []
    historique_gradient = []

    hessienne = hessienne_fonction_cout(matrice_A)

    for k in range(iterations_max):
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)
        norme_gradient = np.linalg.norm(gradient)
        cout = fonction_cout(matrice_A, vecteur_b, x)

        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        if norme_gradient < tolerance:
            break

        direction_newton = -np.linalg.solve(hessienne, gradient)
        x = x + direction_newton

    return x, historique_cout, historique_gradient, k + 1


# fonction qui implémente la méthode de dBFGS pour minimiser la fonction coût
def methode_bfgs(
    matrice_A,
    vecteur_b,
    point_initial,
    pas_initial=1.0,
    tolerance=1e-10,
    iterations_max=300,
):
    x = np.array(point_initial, dtype=float)
    dimension = len(x)

    approximation_inverse_hessienne = np.eye(dimension)

    historique_cout = []
    historique_gradient = []

    for k in range(iterations_max):
        gradient = gradient_fonction_cout(matrice_A, vecteur_b, x)
        norme_gradient = np.linalg.norm(gradient)
        cout = fonction_cout(matrice_A, vecteur_b, x)

        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        if norme_gradient < tolerance:
            break

        direction = -approximation_inverse_hessienne @ gradient

        pas = pas_initial
        cout_courant = cout

        while pas > 1e-12:
            nouveau_x_test = x + pas * direction
            nouveau_cout_test = fonction_cout(matrice_A, vecteur_b, nouveau_x_test)

            if nouveau_cout_test < cout_courant:
                break

            pas *= 0.5

        nouveau_x = x + pas * direction
        nouveau_gradient = gradient_fonction_cout(matrice_A, vecteur_b, nouveau_x)

        delta = nouveau_x - x
        y = nouveau_gradient - gradient
        produit = float(delta @ y)

        if produit > 1e-14:
            Sy = approximation_inverse_hessienne @ y
            terme_1 = (1.0 + (y @ Sy) / produit) * np.outer(delta, delta) / produit
            terme_2 = (np.outer(delta, Sy) + np.outer(Sy, delta)) / produit
            approximation_inverse_hessienne = approximation_inverse_hessienne + terme_1 - terme_2

        x = nouveau_x

    return x, historique_cout, historique_gradient, k + 1


# fonction générique d’ajustement d’un profil par B-spline en utilisant une méthode d’optimisation donnée.
def _ajuster_surface_iterative(points_cible, nb_points_controle, degre, solveur, nom_methode, **kwargs):
    (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base_complete,
    ) = construire_probleme_moindres_carres(points_cible, nb_points_controle, degre)

    x0_interieur = np.linspace(point_initial[0], point_final[0], nb_points_controle)[1:-1]
    y0_interieur = np.linspace(point_initial[1], point_final[1], nb_points_controle)[1:-1]

    solution_x, hist_cout_x, hist_grad_x, nb_iter_x = solveur(
        matrice_interieure, second_membre_x, x0_interieur, **kwargs
    )
    solution_y, hist_cout_y, hist_grad_y, nb_iter_y = solveur(
        matrice_interieure, second_membre_y, y0_interieur, **kwargs
    )

    points_controle = reconstruire_points_controle(point_initial, point_final, solution_x, solution_y)
    courbe_approximee = evaluer_courbe_bspline(points_controle, degre, noeuds)

    approximation_sur_donnees = matrice_base_complete @ points_controle
    erreur_moyenne = np.mean(np.sum((approximation_sur_donnees - points_cible) ** 2, axis=1))

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


# fonction qui ajuste un profil par B-spline en utilisant la méthdoe de descente de gradient
def ajuster_surface_par_gradient(points_cible, nb_points_controle, degre, pas=1e-2):
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


# fonction qui ajuste un profil par B-spline en utilisant la méthdoe de newton 
def ajuster_surface_par_newton(points_cible, nb_points_controle, degre):
    return _ajuster_surface_iterative(
        points_cible,
        nb_points_controle,
        degre,
        solveur=methode_newton,
        nom_methode="Newton",
        tolerance=1e-12,
        iterations_max=50,
    )


# fonction qui ajuste un profil par B-spline en utilisant la méthdoe de BFGS
def ajuster_surface_par_bfgs(points_cible, nb_points_controle, degre):
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