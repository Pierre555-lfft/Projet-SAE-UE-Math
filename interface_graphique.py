#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk  # bibliothèque principale pour créer l'interface graphique
from tkinter import ttk, messagebox  # widgets améliorés + boîtes de dialogue

# import de la fonction qui génère les points du profil NACA
from profils import generer_profil_naca

# import des méthodes directes de résolution
from methodes_directes import (
    ajuster_surface_par_equation_normale,
    ajuster_surface_par_qr,
    ajuster_surface_par_svd,
)

# import des méthodes itératives d’optimisation
from methodes_iteratives import (
    ajuster_surface_par_gradient,
    ajuster_surface_par_newton,
    ajuster_surface_par_bfgs,
)

# import des fonctions d’affichage graphique
from affichage import (
    afficher_profil,
    afficher_convergence_cout,
    afficher_convergence_gradient,
)


# classe principale qui représente l'application graphique complète
class ApplicationOptimisation:

    # constructeur de la classe : appelé automatiquement à la création de l'objet
    def __init__(self, fenetre):
        self.fenetre = fenetre  # stocke la fenêtre principale
        self.fenetre.title("Optimisation géométrique d’un profil NACA")  # titre de la fenêtre
        self.fenetre.geometry("760x520")  # taille initiale de la fenêtre

        # variables qui stockeront les résultats après optimisation
        self.resultat_extrados = None
        self.resultat_intrados = None

        # variables qui stockeront les points du profil généré
        self.points_extrados = None
        self.points_intrados = None

        # crée tous les éléments de l’interface
        self.creer_widgets()

    # crée et place tous les widgets de l’interface
    def creer_widgets(self):
        cadre = ttk.Frame(self.fenetre, padding=15)  # cadre principal avec marges
        cadre.pack(fill="both", expand=True)  # le cadre occupe toute la fenêtre

        # titre principal affiché en haut de l’interface
        titre = ttk.Label(
            cadre,
            text="Optimisation géométrique d’un profil NACA par B-splines",
            font=("Arial", 14, "bold"),
        )
        titre.grid(row=0, column=0, columnspan=2, pady=(0, 15))  # placement dans une grille

        # -----------------------------
        # Champs de saisie des paramètres
        # -----------------------------

        ttk.Label(cadre, text="Cambrure maximale m :").grid(row=1, column=0, sticky="w", pady=5)
        self.entree_m = ttk.Entry(cadre)  # champ de saisie de m
        self.entree_m.insert(0, "0.08")  # valeur par défaut
        self.entree_m.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Position de la cambrure p :").grid(row=2, column=0, sticky="w", pady=5)
        self.entree_p = ttk.Entry(cadre)  # champ de saisie de p
        self.entree_p.insert(0, "0.4")
        self.entree_p.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Épaisseur maximale t :").grid(row=3, column=0, sticky="w", pady=5)
        self.entree_t = ttk.Entry(cadre)  # champ de saisie de t
        self.entree_t.insert(0, "0.16")
        self.entree_t.grid(row=3, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Nombre de points du profil :").grid(row=4, column=0, sticky="w", pady=5)
        self.entree_nb_points = ttk.Entry(cadre)  # nombre de points du profil cible
        self.entree_nb_points.insert(0, "120")
        self.entree_nb_points.grid(row=4, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Nombre de points de contrôle :").grid(row=5, column=0, sticky="w", pady=5)
        self.entree_nb_ctrl = ttk.Entry(cadre)  # nombre de points de contrôle de la B-spline
        self.entree_nb_ctrl.insert(0, "10")
        self.entree_nb_ctrl.grid(row=5, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Degré de la B-spline :").grid(row=6, column=0, sticky="w", pady=5)
        self.entree_degre = ttk.Entry(cadre)  # degré de la B-spline
        self.entree_degre.insert(0, "3")
        self.entree_degre.grid(row=6, column=1, sticky="ew", pady=5)

        # liste déroulante pour choisir la méthode d’optimisation
        ttk.Label(cadre, text="Méthode :").grid(row=7, column=0, sticky="w", pady=5)
        self.choix_methode = ttk.Combobox(
            cadre,
            state="readonly",  # empêche l'utilisateur d'écrire autre chose que les choix proposés
            values=[
                "Équation normale",
                "QR",
                "SVD",
                "Descente de gradient",
                "Newton",
                "BFGS",
            ],
        )
        self.choix_methode.current(1)  # choix par défaut = QR
        self.choix_methode.grid(row=7, column=1, sticky="ew", pady=5)

        # champ supplémentaire utile uniquement pour la descente de gradient
        ttk.Label(cadre, text="Pas du gradient (si besoin) :").grid(row=8, column=0, sticky="w", pady=5)
        self.entree_pas = ttk.Entry(cadre)
        self.entree_pas.insert(0, "0.01")
        self.entree_pas.grid(row=8, column=1, sticky="ew", pady=5)

        # -----------------------------
        # Boutons de commande
        # -----------------------------

        cadre_boutons = ttk.Frame(cadre)  # sous-cadre réservé aux boutons
        cadre_boutons.grid(row=9, column=0, columnspan=2, pady=20)

        # bouton qui lance le calcul d’optimisation
        bouton_lancer = ttk.Button(
            cadre_boutons,
            text="Lancer l’optimisation",
            command=self.lancer_optimisation,  # appelle la méthode correspondante
        )
        bouton_lancer.grid(row=0, column=0, padx=5, pady=5)

        # bouton qui affiche le profil optimisé
        bouton_afficher = ttk.Button(
            cadre_boutons,
            text="Afficher le profil",
            command=self.afficher_resultat,
        )
        bouton_afficher.grid(row=0, column=1, padx=5, pady=5)

        # bouton qui affiche la convergence du coût
        bouton_cout = ttk.Button(
            cadre_boutons,
            text="Convergence du coût",
            command=self.afficher_convergence_cout_resultat,
        )
        bouton_cout.grid(row=0, column=2, padx=5, pady=5)

        # bouton qui affiche la convergence du gradient
        bouton_gradient = ttk.Button(
            cadre_boutons,
            text="Convergence du gradient",
            command=self.afficher_convergence_gradient_resultat,
        )
        bouton_gradient.grid(row=0, column=3, padx=5, pady=5)

        # bouton qui ferme l’application
        bouton_quitter = ttk.Button(
            cadre_boutons,
            text="Quitter",
            command=self.fenetre.destroy,
        )
        bouton_quitter.grid(row=0, column=4, padx=5, pady=5)

        # -----------------------------
        # Zone d’affichage des résultats texte
        # -----------------------------

        ttk.Label(cadre, text="Résultats :").grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 5))

        self.zone_resultats = tk.Text(cadre, height=10, width=80)  # zone multiligne pour afficher les résultats
        self.zone_resultats.grid(row=11, column=0, columnspan=2, sticky="nsew")

        # permet à la colonne de droite et à la zone de résultats de s’étendre quand la fenêtre est redimensionnée
        cadre.columnconfigure(1, weight=1)
        cadre.rowconfigure(11, weight=1)

    # lit les valeurs saisies dans l’interface et les convertit en nombres
    def lire_parametres(self):
        try:
            m = float(self.entree_m.get())  # lit la cambrure maximale
            p = float(self.entree_p.get())  # lit la position de la cambrure
            t = float(self.entree_t.get())  # lit l’épaisseur maximale
            nb_points = int(self.entree_nb_points.get())  # lit le nombre de points du profil
            nb_points_controle = int(self.entree_nb_ctrl.get())  # lit le nombre de points de contrôle
            degre = int(self.entree_degre.get())  # lit le degré de la B-spline
            pas = float(self.entree_pas.get())  # lit le pas du gradient
        except ValueError:
            # erreur si l’utilisateur entre un texte non convertible
            raise ValueError("Un ou plusieurs paramètres sont invalides.")

        # renvoie tous les paramètres dans un tuple
        return m, p, t, nb_points, nb_points_controle, degre, pas

    # vérifie si la méthode utilisée possède des historiques de convergence
    def methode_iterative_disponible(self):
        return (
            self.resultat_extrados is not None
            and self.resultat_intrados is not None
            and "historique_cout_y" in self.resultat_extrados
            and "historique_gradient_y" in self.resultat_extrados
        )

    # lance toute la procédure d’optimisation
    def lancer_optimisation(self):
        try:
            # récupère les paramètres saisis par l’utilisateur
            m, p, t, nb_points, nb_points_controle, degre, pas = self.lire_parametres()

            # génère les points du profil NACA cible
            self.points_extrados, self.points_intrados = generer_profil_naca(
                m, p, t, nb_points=nb_points
            )

            # lit la méthode choisie dans la liste déroulante
            methode = self.choix_methode.get()

            # selon la méthode choisie, appelle la bonne fonction d’optimisation
            if methode == "Équation normale":
                self.resultat_extrados = ajuster_surface_par_equation_normale(
                    self.points_extrados, nb_points_controle, degre
                )
                self.resultat_intrados = ajuster_surface_par_equation_normale(
                    self.points_intrados, nb_points_controle, degre
                )

            elif methode == "QR":
                self.resultat_extrados = ajuster_surface_par_qr(
                    self.points_extrados, nb_points_controle, degre
                )
                self.resultat_intrados = ajuster_surface_par_qr(
                    self.points_intrados, nb_points_controle, degre
                )

            elif methode == "SVD":
                self.resultat_extrados = ajuster_surface_par_svd(
                    self.points_extrados, nb_points_controle, degre
                )
                self.resultat_intrados = ajuster_surface_par_svd(
                    self.points_intrados, nb_points_controle, degre
                )

            elif methode == "Descente de gradient":
                self.resultat_extrados = ajuster_surface_par_gradient(
                    self.points_extrados, nb_points_controle, degre, pas=pas
                )
                self.resultat_intrados = ajuster_surface_par_gradient(
                    self.points_intrados, nb_points_controle, degre, pas=pas
                )

            elif methode == "Newton":
                self.resultat_extrados = ajuster_surface_par_newton(
                    self.points_extrados, nb_points_controle, degre
                )
                self.resultat_intrados = ajuster_surface_par_newton(
                    self.points_intrados, nb_points_controle, degre
                )

            elif methode == "BFGS":
                self.resultat_extrados = ajuster_surface_par_bfgs(
                    self.points_extrados, nb_points_controle, degre
                )
                self.resultat_intrados = ajuster_surface_par_bfgs(
                    self.points_intrados, nb_points_controle, degre
                )

            else:
                # sécurité si aucune méthode valide n’a été sélectionnée
                raise ValueError("Méthode inconnue.")

            # calcule l’erreur totale comme somme des erreurs extrados + intrados
            erreur_totale = (
                self.resultat_extrados["erreur_moyenne"]
                + self.resultat_intrados["erreur_moyenne"]
            )

            # vide la zone texte avant d'afficher les nouveaux résultats
            self.zone_resultats.delete("1.0", tk.END)

            # affiche les résultats principaux dans la zone texte
            self.zone_resultats.insert(tk.END, f"Méthode choisie : {self.resultat_extrados['nom_methode']}\n")
            self.zone_resultats.insert(tk.END, f"Erreur moyenne extrados : {self.resultat_extrados['erreur_moyenne']:.6e}\n")
            self.zone_resultats.insert(tk.END, f"Erreur moyenne intrados : {self.resultat_intrados['erreur_moyenne']:.6e}\n")
            self.zone_resultats.insert(tk.END, f"Erreur totale : {erreur_totale:.6e}\n")

            # si la méthode est itérative, affiche aussi le nombre d’itérations
            if "nombre_iterations_x" in self.resultat_extrados:
                self.zone_resultats.insert(
                    tk.END,
                    f"Itérations extrados (x, y) : "
                    f"{self.resultat_extrados['nombre_iterations_x']}, "
                    f"{self.resultat_extrados['nombre_iterations_y']}\n"
                )
                self.zone_resultats.insert(
                    tk.END,
                    f"Itérations intrados (x, y) : "
                    f"{self.resultat_intrados['nombre_iterations_x']}, "
                    f"{self.resultat_intrados['nombre_iterations_y']}\n"
                )

            # affiche une boîte de dialogue de succès
            messagebox.showinfo("Succès", "Optimisation terminée.")

        except Exception as e:
            # en cas d'erreur, affiche le message dans une boîte de dialogue
            messagebox.showerror("Erreur", str(e))

    # affiche le profil initial et le profil approché
    def afficher_resultat(self):
        # empêche l’affichage si aucune optimisation n’a encore été lancée
        if self.resultat_extrados is None or self.resultat_intrados is None:
            messagebox.showwarning("Attention", "Lance d’abord une optimisation.")
            return

        # appelle la fonction d’affichage du profil
        afficher_profil(
            self.points_extrados,
            self.points_intrados,
            self.resultat_extrados,
            self.resultat_intrados,
        )

    # affiche la courbe de convergence du coût
    def afficher_convergence_cout_resultat(self):
        # vérifie qu’une optimisation existe
        if self.resultat_extrados is None or self.resultat_intrados is None:
            messagebox.showwarning("Attention", "Lance d’abord une optimisation.")
            return

        # interdit l’affichage pour les méthodes directes
        if not self.methode_iterative_disponible():
            messagebox.showinfo(
                "Information",
                "La convergence du coût n’est disponible que pour les méthodes itératives."
            )
            return

        # affiche le graphe du coût
        afficher_convergence_cout(self.resultat_extrados, self.resultat_intrados)

    # affiche la courbe de convergence du gradient
    def afficher_convergence_gradient_resultat(self):
        # vérifie qu’une optimisation existe
        if self.resultat_extrados is None or self.resultat_intrados is None:
            messagebox.showwarning("Attention", "Lance d’abord une optimisation.")
            return

        # interdit l’affichage pour les méthodes directes
        if not self.methode_iterative_disponible():
            messagebox.showinfo(
                "Information",
                "La convergence du gradient n’est disponible que pour les méthodes itératives."
            )
            return

        # affiche le graphe du gradient
        afficher_convergence_gradient(self.resultat_extrados, self.resultat_intrados)


# point d’entrée du programme
if __name__ == "__main__":
    racine = tk.Tk()  # crée la fenêtre principale Tkinter
    app = ApplicationOptimisation(racine)  # crée l’objet application
    racine.mainloop()  # lance la boucle principale de l’interface