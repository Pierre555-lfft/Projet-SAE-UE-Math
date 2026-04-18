

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from profils import generer_profil_naca
from methodes_directes import (
    ajuster_surface_par_equation_normale,
    ajuster_surface_par_qr,
    ajuster_surface_par_svd,
)
from methodes_iteratives import (
    ajuster_surface_par_gradient,
    ajuster_surface_par_newton,
    ajuster_surface_par_bfgs,
)
from affichage import afficher_profil, afficher_convergence


# fonction qui demande à l'utilisateur de saisir un nombre réel 
def demander_flottant(message, valeur_defaut):
    texte = input(f"{message} [{valeur_defaut}] : ").strip()
    if texte == "":
        return float(valeur_defaut)
    return float(texte)


# fonction qui demande à l'utilisateur de saisir un nombre entier
def demander_entier(message, valeur_defaut):
    texte = input(f"{message} [{valeur_defaut}] : ").strip()
    if texte == "":
        return int(valeur_defaut)
    return int(texte)

# fonction qui affiche un menu à l'utilisateur
def choisir_methode():
    print("\nChoisir une méthode :")
    print("1 - Équation normale")
    print("2 - QR")
    print("3 - SVD")
    print("4 - Descente de gradient")
    print("5 - Newton")
    print("6 - BFGS")
    return input("Votre choix : ").strip()


# fonction principale d'optimisation géométrique d'un profil NACA
def main():
    print("=== Optimisation géométrique d'un profil NACA par B-splines ===")

    m = demander_flottant("Cambrure maximale m", 0.08)
    p = demander_flottant("Position de la cambrure p", 0.4)
    t = demander_flottant("Épaisseur maximale t", 0.16)
    nb_points = demander_entier("Nombre de points du profil", 120)
    nb_points_controle = demander_entier("Nombre de points de contrôle", 10)
    degre = demander_entier("Degré de la B-spline", 3)

    choix = choisir_methode()

    points_extrados, points_intrados = generer_profil_naca(m, p, t, nb_points=nb_points)

    if choix == "1":
        resultat_extrados = ajuster_surface_par_equation_normale(points_extrados, nb_points_controle, degre)
        resultat_intrados = ajuster_surface_par_equation_normale(points_intrados, nb_points_controle, degre)

    elif choix == "2":
        resultat_extrados = ajuster_surface_par_qr(points_extrados, nb_points_controle, degre)
        resultat_intrados = ajuster_surface_par_qr(points_intrados, nb_points_controle, degre)

    elif choix == "3":
        resultat_extrados = ajuster_surface_par_svd(points_extrados, nb_points_controle, degre)
        resultat_intrados = ajuster_surface_par_svd(points_intrados, nb_points_controle, degre)

    elif choix == "4":
        pas = demander_flottant("Pas de gradient", 1e-2)
        resultat_extrados = ajuster_surface_par_gradient(points_extrados, nb_points_controle, degre, pas=pas)
        resultat_intrados = ajuster_surface_par_gradient(points_intrados, nb_points_controle, degre, pas=pas)

    elif choix == "5":
        resultat_extrados = ajuster_surface_par_newton(points_extrados, nb_points_controle, degre)
        resultat_intrados = ajuster_surface_par_newton(points_intrados, nb_points_controle, degre)

    elif choix == "6":
        resultat_extrados = ajuster_surface_par_bfgs(points_extrados, nb_points_controle, degre)
        resultat_intrados = ajuster_surface_par_bfgs(points_intrados, nb_points_controle, degre)

    else:
        print("Choix invalide.")
        return

    erreur_totale = resultat_extrados["erreur_moyenne"] + resultat_intrados["erreur_moyenne"]

    print("\n=== Résultats ===")
    print(f"Méthode choisie         : {resultat_extrados['nom_methode']}")
    print(f"Erreur moyenne extrados : {resultat_extrados['erreur_moyenne']:.6e}")
    print(f"Erreur moyenne intrados : {resultat_intrados['erreur_moyenne']:.6e}")
    print(f"Erreur totale           : {erreur_totale:.6e}")

    if "nombre_iterations_x" in resultat_extrados:
        print(
            f"Itérations extrados (x, y) : "
            f"{resultat_extrados['nombre_iterations_x']}, {resultat_extrados['nombre_iterations_y']}"
        )
        print(
            f"Itérations intrados (x, y) : "
            f"{resultat_intrados['nombre_iterations_x']}, {resultat_intrados['nombre_iterations_y']}"
        )

    afficher_profil(points_extrados, points_intrados, resultat_extrados, resultat_intrados)
    afficher_convergence(resultat_extrados, resultat_intrados)


if __name__ == "__main__":
    main()