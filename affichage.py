#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt  # bibliothèque utilisée pour tracer les graphiques



# fonction qui affiche la comparaison entre le profil cible et
# le profil reconstruit par B-spline
def afficher_profil(points_extrados, points_intrados, resultat_extrados, resultat_intrados):

    # crée une nouvelle fenêtre de figure
    plt.figure(figsize=(10, 5))

    # trace les points du profil cible pour l’extrados
    plt.plot(
        points_extrados[:, 0],          # abscisses x des points extrados
        points_extrados[:, 1],          # ordonnées y des points extrados
        "r--",                          # rouge pointillé
        label="Extrados cible"
    )

    # trace les points du profil cible pour l’intrados
    plt.plot(
        points_intrados[:, 0],          # abscisses x des points intrados
        points_intrados[:, 1],          # ordonnées y des points intrados
        "m--",                          # magenta pointillé
        label="Intrados cible"
    )

    # trace la courbe B-spline reconstruite pour l’extrados
    plt.plot(
        resultat_extrados["courbe"][:, 0],   # abscisses de la courbe approchée
        resultat_extrados["courbe"][:, 1],   # ordonnées de la courbe approchée
        "b",                                 # bleu
        linewidth=2,                         # épaisseur de trait
        label="Extrados B-spline",
    )

    # trace la courbe B-spline reconstruite pour l’intrados
    plt.plot(
        resultat_intrados["courbe"][:, 0],
        resultat_intrados["courbe"][:, 1],
        "g",                                 # vert
        linewidth=2,
        label="Intrados B-spline",
    )

    # affiche les points de contrôle de l’extrados
    plt.scatter(
        resultat_extrados["points_controle"][:, 0],  # abscisses des points de contrôle
        resultat_extrados["points_controle"][:, 1],  # ordonnées des points de contrôle
        c="black",                                   # couleur noire
        s=20,                                        # taille des marqueurs
        label="Points de contrôle extrados",
    )

    # affiche les points de contrôle de l’intrados
    plt.scatter(
        resultat_intrados["points_controle"][:, 0],
        resultat_intrados["points_controle"][:, 1],
        c="orange",                                  # couleur orange
        s=20,
        label="Points de contrôle intrados",
    )

    # impose la même échelle sur x et y pour ne pas déformer visuellement le profil
    plt.axis("equal")

    # active la grille
    plt.grid(True)

    # nom des axes
    plt.xlabel("x")
    plt.ylabel("y")

    # titre du graphique avec le nom de la méthode utilisée
    plt.title(
        f"Approximation géométrique par {resultat_extrados['nom_methode']}"
    )

    # affiche la légende
    plt.legend()

    # ajuste automatiquement la mise en page
    plt.tight_layout()

    # affiche la figure
    plt.show()



# fonction qui affiche uniquement la convergence du coût
def afficher_convergence_cout(resultat_extrados, resultat_intrados):

    # vérifie que les historiques de coût existent
    # si la méthode est directe, il n'y a pas d'historique
    if "historique_cout_y" not in resultat_extrados:
        return

    # crée une nouvelle figure pour la convergence du coût
    plt.figure(figsize=(10, 4))

    # trace l'évolution du coût pour l’extrados
    plt.plot(
        resultat_extrados["historique_cout_y"],
        label="Coût extrados"
    )

    # trace l'évolution du coût pour l’intrados
    plt.plot(
        resultat_intrados["historique_cout_y"],
        label="Coût intrados"
    )

    # met l’axe vertical en échelle logarithmique
    # pour mieux visualiser la décroissance du coût
    plt.yscale("log")

    # active la grille
    plt.grid(True)

    # labels des axes
    plt.xlabel("Itération")
    plt.ylabel("Fonction coût")

    # titre du graphique
    plt.title(
        f"Convergence du coût ({resultat_extrados['nom_methode']})"
    )

    # affiche la légende
    plt.legend()

    # ajuste la mise en page
    plt.tight_layout()

    # affiche la figure
    plt.show()



# fonction qui affiche uniquement la convergence du gradient
def afficher_convergence_gradient(resultat_extrados, resultat_intrados):

    # vérifie que les historiques du gradient existent
    if "historique_gradient_y" not in resultat_extrados:
        return

    # crée une nouvelle figure pour la convergence du gradient
    plt.figure(figsize=(10, 4))

    # trace l'évolution de la norme du gradient pour l’extrados
    plt.plot(
        resultat_extrados["historique_gradient_y"],
        label="||grad|| extrados"
    )

    # trace l'évolution de la norme du gradient pour l’intrados
    plt.plot(
        resultat_intrados["historique_gradient_y"],
        label="||grad|| intrados"
    )

    # met l’axe vertical en échelle logarithmique
    # pour voir plus clairement la diminution du gradient
    plt.yscale("log")

    # active la grille
    plt.grid(True)

    # labels des axes
    plt.xlabel("Itération")
    plt.ylabel("Norme du gradient")

    # titre du graphique
    plt.title(
        f"Convergence du gradient ({resultat_extrados['nom_methode']})"
    )

    # affiche la légende
    plt.legend()

    # ajuste la mise en page
    plt.tight_layout()

    # affiche la figure
    plt.show()