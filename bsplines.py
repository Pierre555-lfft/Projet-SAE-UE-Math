#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  # importe NumPy pour les calculs matriciels et vectoriels



# fonction qui calcule une fonction de base de B-spline N_{i,p}(u)
# avec la formule récursive de Cox-De Boor
def base_bspline(indice, degre, noeuds, u):

    # cas de base : degré 0
    # la fonction vaut 1 sur un intervalle de noeuds, 0 ailleurs
    if degre == 0:

        # on vérifie si u appartient à l'intervalle [t_i, t_{i+1}[
        # le second test gère le cas particulier u = 1.0
        if (noeuds[indice] <= u < noeuds[indice + 1]) or (
            abs(u - 1.0) < 1e-14 and abs(noeuds[indice + 1] - 1.0) < 1e-14
        ):
            return 1.0

        # si u n'est pas dans l'intervalle, la fonction vaut 0
        return 0.0

    # initialisation des deux termes de la formule récursive
    terme_gauche = 0.0
    terme_droite = 0.0

    # dénominateur du premier terme de Cox-De Boor
    denominateur_1 = noeuds[indice + degre] - noeuds[indice]

    # dénominateur du second terme de Cox-De Boor
    denominateur_2 = noeuds[indice + degre + 1] - noeuds[indice + 1]

    # calcul du premier terme si le dénominateur n'est pas nul
    if abs(denominateur_1) > 1e-14:
        terme_gauche = (
            (u - noeuds[indice]) / denominateur_1
        ) * base_bspline(
            indice, degre - 1, noeuds, u
        )

    # calcul du second terme si le dénominateur n'est pas nul
    if abs(denominateur_2) > 1e-14:
        terme_droite = (
            (noeuds[indice + degre + 1] - u) / denominateur_2
        ) * base_bspline(
            indice + 1, degre - 1, noeuds, u
        )

    # la fonction de base de degré p est la somme des deux termes
    return terme_gauche + terme_droite



# fonction qui construit la matrice A des fonctions de base
# chaque coefficient A[i,j] = N_{j,p}(u_i)
def construire_matrice_base(valeurs_u, nb_points_controle, degre, noeuds):

    # crée une matrice nulle de taille :
    # nombre de paramètres u  x  nombre de points de contrôle
    matrice = np.zeros((len(valeurs_u), nb_points_controle), dtype=float)

    # boucle sur toutes les valeurs du paramètre u
    for i, u in enumerate(valeurs_u):

        # boucle sur toutes les fonctions de base
        for j in range(nb_points_controle):

            # calcule la valeur de la j-ième fonction de base au point u
            matrice[i, j] = base_bspline(j, degre, noeuds, float(u))

    # renvoie la matrice complète des fonctions de base
    return matrice



# fonction construit un vecteur de noeuds ouverts uniformes
def construire_noeuds_ouverts_uniformes(nb_points_controle, degre):

    # nombre total de noeuds pour une B-spline
    nb_noeuds = nb_points_controle + degre + 1

    # initialisation du vecteur de noeuds
    noeuds = np.zeros(nb_noeuds)

    # répète le noeud 0 au début (degré + 1 fois)
    # cela force la courbe à passer par le premier point de contrôle
    noeuds[:degre + 1] = 0.0

    # répète le noeud 1 à la fin (degré + 1 fois)
    # cela force la courbe à passer par le dernier point de contrôle
    noeuds[-(degre + 1):] = 1.0

    # nombre de noeuds internes non répétés
    nb_noeuds_internes = nb_noeuds - 2 * (degre + 1)

    # si des noeuds internes existent, on les répartit uniformément
    if nb_noeuds_internes > 0:
        noeuds_internes = np.linspace(0.0, 1.0, nb_noeuds_internes + 2)[1:-1]
        noeuds[degre + 1:-(degre + 1)] = noeuds_internes

    # renvoie le vecteur de noeuds complet
    return noeuds



# fonction qui construit le problème de moindres carrés AX ≈ b
# pour approximer un nuage de points par une B-spline
def construire_probleme_moindres_carres(points_cible, nb_points_controle, degre):

    # nombre de points du profil cible
    nb_points = len(points_cible)

    # discrétisation uniforme du paramètre u entre 0 et 1
    valeurs_u = np.linspace(0.0, 1.0, nb_points)

    # construit les noeuds de la B-spline
    noeuds = construire_noeuds_ouverts_uniformes(nb_points_controle, degre)

    # construit la matrice des fonctions de base
    matrice_base = construire_matrice_base(valeurs_u, nb_points_controle, degre, noeuds)

    # première colonne de la matrice de base
    # correspond au premier point de contrôle fixé
    c0 = matrice_base[:, 0]

    # dernière colonne de la matrice de base
    # correspond au dernier point de contrôle fixé
    cn = matrice_base[:, -1]

    # matrice intérieure : colonnes associées aux points de contrôle libres
    matrice_interieure = matrice_base[:, 1:-1]

    # premier point du nuage cible
    point_initial = points_cible[0].copy()

    # dernier point du nuage cible
    point_final = points_cible[-1].copy()

    # second membre du problème pour les coordonnées x
    # on enlève la contribution des points de contrôle fixés
    second_membre_x = (
        points_cible[:, 0]
        - point_initial[0] * c0
        - point_final[0] * cn
    )

    # second membre du problème pour les coordonnées y
    second_membre_y = (
        points_cible[:, 1]
        - point_initial[1] * c0
        - point_final[1] * cn
    )

    # renvoie tous les éléments utiles à la résolution
    return (
        matrice_interieure,
        second_membre_x,
        second_membre_y,
        point_initial,
        point_final,
        noeuds,
        matrice_base,
    )



# fonction qui reconstruit l’ensemble des points de contrôle après optimisation
def reconstruire_points_controle(point_initial, point_final, x_interieur, y_interieur):

    # nombre total de points de contrôle :
    # points intérieurs + 2 extrémités
    nb_points_controle = len(x_interieur) + 2

    # matrice qui contiendra tous les points de contrôle (x,y)
    points_controle = np.zeros((nb_points_controle, 2), dtype=float)

    # impose le premier point de contrôle
    points_controle[0] = point_initial

    # impose le dernier point de contrôle
    points_controle[-1] = point_final

    # remplit les coordonnées x des points intérieurs
    points_controle[1:-1, 0] = x_interieur

    # remplit les coordonnées y des points intérieurs
    points_controle[1:-1, 1] = y_interieur

    # renvoie l'ensemble des points de contrôle
    return points_controle


# fonction qui évalue la courbe B-spline finale pour pouvoir l’afficher
def evaluer_courbe_bspline(points_controle, degre, noeuds, nb_points=400):

    # discrétisation fine du paramètre u pour tracer la courbe
    valeurs_u = np.linspace(0.0, 1.0, nb_points)

    # construction de la matrice des fonctions de base sur cette discrétisation
    matrice = construire_matrice_base(valeurs_u, len(points_controle), degre, noeuds)

    # calcul des points de la courbe :
    # P = A * points_controle
    return matrice @ points_controle