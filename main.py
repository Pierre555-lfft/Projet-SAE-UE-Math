#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import de la fonction qui génère le profil NACA cible
from profils import generer_profil_naca


# import des méthodes directes
from methodes_directes import (
    ajuster_surface_par_equation_normale,
    ajuster_surface_par_qr,
    ajuster_surface_par_svd,
)


# import des méthodes itératives
from methodes_iteratives import (
    ajuster_surface_par_gradient,
    ajuster_surface_par_newton,
    ajuster_surface_par_bfgs,
)


# import des fonctions d'affichage
from affichage import (
    afficher_profil,
    afficher_convergence_cout,
    afficher_convergence_gradient
)



# fonction qui demande à l'utilisateur un nombre réel
# si rien n'est saisi -> prend la valeur par défaut
def demander_flottant(message, valeur_defaut):

    # affiche le message et récupère la saisie utilisateur
    texte = input(
        f"{message} [{valeur_defaut}] : "
    ).strip()

    # si entrée vide -> valeur par défaut
    if texte == "":
        return float(valeur_defaut)

    # sinon conversion en réel
    return float(texte)



# fonction qui demande un entier à l'utilisateur
def demander_entier(message, valeur_defaut):

    texte = input(
        f"{message} [{valeur_defaut}] : "
    ).strip()

    if texte == "":
        return int(valeur_defaut)

    return int(texte)



# fonction qui affiche le menu des méthodes disponibles
# et récupère le choix utilisateur
def choisir_methode():

    print("\nChoisir une méthode :")

    print("1 - Équation normale")
    print("2 - QR")
    print("3 - SVD")
    print("4 - Descente de gradient")
    print("5 - Newton")
    print("6 - BFGS")

    # renvoie le choix utilisateur
    return input(
        "Votre choix : "
    ).strip()


# programme principal
def main():

    # message d'accueil
    print(
        "=== Optimisation géométrique d'un profil NACA par B-splines ==="
    )


    
    # lecture des paramètres du profil

    m = demander_flottant("Cambrure maximale m",0.08)

    p = demander_flottant("Position de la cambrure p",0.4)

    t = demander_flottant("Épaisseur maximale t",0.16)

    nb_points = demander_entier("Nombre de points du profil",120)

    nb_points_controle = demander_entier("Nombre de points de contrôle",10)

    degre = demander_entier("Degré de la B-spline",3)


    # lecture du choix de méthode
    choix = choisir_methode()


    
    # génération du profil NACA
    

    points_extrados, points_intrados = generer_profil_naca(m,p,t,nb_points=nb_points)


    
    # choix de la méthode de résolution
    

    if choix == "1":

        # résolution par équation normale
        resultat_extrados = ajuster_surface_par_equation_normale(points_extrados,nb_points_controle,degre)
        resultat_intrados = ajuster_surface_par_equation_normale(points_intrados,nb_points_controle,degre)


    elif choix == "2":

        # résolution par QR
        resultat_extrados = ajuster_surface_par_qr(points_extrados,nb_points_controle,degre)
        resultat_intrados = ajuster_surface_par_qr(points_intrados,nb_points_controle,degre)


    elif choix == "3":

        # résolution par SVD
        resultat_extrados = ajuster_surface_par_svd(points_extrados,nb_points_controle,degre)
        resultat_intrados = ajuster_surface_par_svd(points_intrados,nb_points_controle,degre)


    elif choix == "4":

        # lecture du pas de gradient
        pas = demander_flottant(
            "Pas de gradient",
            1e-2
        )

        # résolution par descente de gradient
        resultat_extrados = ajuster_surface_par_gradient(points_extrados,nb_points_controle,degre,pas=pas)
        resultat_intrados = ajuster_surface_par_gradient(points_intrados,nb_points_controle,degre,pas=pas)


    elif choix == "5":

        # résolution par Newton
        resultat_extrados = ajuster_surface_par_newton(points_extrados,nb_points_controle,degre)
        resultat_intrados = ajuster_surface_par_newton(points_intrados,nb_points_controle,degre)


    elif choix == "6":

        # résolution par BFGS
        resultat_extrados = ajuster_surface_par_bfgs(points_extrados,nb_points_controle,degre)
        resultat_intrados = ajuster_surface_par_bfgs(points_intrados,nb_points_controle,degre)


    else:

        # sécurité si choix invalide
        print(
            "Choix invalide."
        )

        return


    # -----------------------------------
    # calcul erreur globale
    # -----------------------------------

    erreur_totale = (
        resultat_extrados["erreur_moyenne"]
        +
        resultat_intrados["erreur_moyenne"]
    )


    # -----------------------------------
    # affichage des résultats numériques
    # -----------------------------------

    print("\n=== Résultats ===")

    print(
        f"Méthode choisie : "
        f"{resultat_extrados['nom_methode']}"
    )

    print(
        f"Erreur moyenne extrados : "
        f"{resultat_extrados['erreur_moyenne']:.6e}"
    )

    print(
        f"Erreur moyenne intrados : "
        f"{resultat_intrados['erreur_moyenne']:.6e}"
    )

    print(
        f"Erreur totale : "
        f"{erreur_totale:.6e}"
    )


    # si méthode itérative :
    # afficher nombre d'itérations
    if "nombre_iterations_x" in resultat_extrados:

        print(
            f"Itérations extrados (x,y) : "
            f"{resultat_extrados['nombre_iterations_x']}, "
            f"{resultat_extrados['nombre_iterations_y']}"
        )

        print(
            f"Itérations intrados (x,y) : "
            f"{resultat_intrados['nombre_iterations_x']}, "
            f"{resultat_intrados['nombre_iterations_y']}"
        )


    # -----------------------------------
    # affichage graphique du profil
    # -----------------------------------

    afficher_profil(
        points_extrados,
        points_intrados,
        resultat_extrados,
        resultat_intrados
    )


    # -----------------------------------
    # affichage convergence si méthode itérative
    # -----------------------------------

    if "historique_cout_y" in resultat_extrados:

        afficher_convergence_cout(
            resultat_extrados,
            resultat_intrados
        )

        afficher_convergence_gradient(
            resultat_extrados,
            resultat_intrados
        )

    else:

        print(
            "Pas de courbe de convergence "
            "pour cette méthode directe"
        )


# ==========================================================
# point d'entrée du programme
# ==========================================================
if __name__ == "__main__":

    # lance le programme principal
    main()