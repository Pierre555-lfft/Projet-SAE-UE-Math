#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


# fonction qui affiche la comparaison entre le profil cible et l’approximation B-spline
def afficher_profil(points_extrados, points_intrados, resultat_extrados, resultat_intrados):
    plt.figure(figsize=(10, 5))

    plt.plot(points_extrados[:, 0], points_extrados[:, 1], "r--", label="Extrados cible")
    plt.plot(points_intrados[:, 0], points_intrados[:, 1], "m--", label="Intrados cible")

    plt.plot(
        resultat_extrados["courbe"][:, 0],
        resultat_extrados["courbe"][:, 1],
        "b",
        linewidth=2,
        label="Extrados B-spline",
    )
    plt.plot(
        resultat_intrados["courbe"][:, 0],
        resultat_intrados["courbe"][:, 1],
        "g",
        linewidth=2,
        label="Intrados B-spline",
    )

    plt.scatter(
        resultat_extrados["points_controle"][:, 0],
        resultat_extrados["points_controle"][:, 1],
        c="black",
        s=20,
        label="Points de contrôle extrados",
    )
    plt.scatter(
        resultat_intrados["points_controle"][:, 0],
        resultat_intrados["points_controle"][:, 1],
        c="orange",
        s=20,
        label="Points de contrôle intrados",
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Approximation géométrique par {resultat_extrados['nom_methode']}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# fonction qui affiche uniquement la convergence du coût
def afficher_convergence_cout(resultat_extrados, resultat_intrados):
    if "historique_cout_y" not in resultat_extrados:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(resultat_extrados["historique_cout_y"], label="Coût extrados")
    plt.plot(resultat_intrados["historique_cout_y"], label="Coût intrados")
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Itération")
    plt.ylabel("Fonction coût")
    plt.title(f"Convergence du coût ({resultat_extrados['nom_methode']})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# fonction qui affiche uniquement la convergence du gradient
def afficher_convergence_gradient(resultat_extrados, resultat_intrados):
    if "historique_gradient_y" not in resultat_extrados:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(resultat_extrados["historique_gradient_y"], label="||grad|| extrados")
    plt.plot(resultat_intrados["historique_gradient_y"], label="||grad|| intrados")
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Itération")
    plt.ylabel("Norme du gradient")
    plt.title(f"Convergence du gradient ({resultat_extrados['nom_methode']})")
    plt.legend()
    plt.tight_layout()
    plt.show()