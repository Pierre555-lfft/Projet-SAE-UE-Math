

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox

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


# application graphique
class ApplicationOptimisation:
    # fonction qui initialisation l'application graphique
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.fenetre.title("Optimisation géométrique d’un profil NACA")
        self.fenetre.geometry("650x520")

        self.resultat_extrados = None
        self.resultat_intrados = None
        self.points_extrados = None
        self.points_intrados = None

        self.creer_widgets()
    
    # fonction qui crée et organise les éléments graphique de l'interface 
    def creer_widgets(self):
        cadre = ttk.Frame(self.fenetre, padding=15)
        cadre.pack(fill="both", expand=True)

        titre = ttk.Label(
            cadre,
            text="Optimisation géométrique d’un profil NACA par B-splines",
            font=("Arial", 14, "bold"),
        )
        titre.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        # Paramètres NACA
        ttk.Label(cadre, text="Cambrure maximale m :").grid(row=1, column=0, sticky="w", pady=5)
        self.entree_m = ttk.Entry(cadre)
        self.entree_m.insert(0, "0.08")
        self.entree_m.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Position de la cambrure p :").grid(row=2, column=0, sticky="w", pady=5)
        self.entree_p = ttk.Entry(cadre)
        self.entree_p.insert(0, "0.4")
        self.entree_p.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Épaisseur maximale t :").grid(row=3, column=0, sticky="w", pady=5)
        self.entree_t = ttk.Entry(cadre)
        self.entree_t.insert(0, "0.16")
        self.entree_t.grid(row=3, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Nombre de points du profil :").grid(row=4, column=0, sticky="w", pady=5)
        self.entree_nb_points = ttk.Entry(cadre)
        self.entree_nb_points.insert(0, "120")
        self.entree_nb_points.grid(row=4, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Nombre de points de contrôle :").grid(row=5, column=0, sticky="w", pady=5)
        self.entree_nb_ctrl = ttk.Entry(cadre)
        self.entree_nb_ctrl.insert(0, "10")
        self.entree_nb_ctrl.grid(row=5, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Degré de la B-spline :").grid(row=6, column=0, sticky="w", pady=5)
        self.entree_degre = ttk.Entry(cadre)
        self.entree_degre.insert(0, "3")
        self.entree_degre.grid(row=6, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Méthode :").grid(row=7, column=0, sticky="w", pady=5)
        self.choix_methode = ttk.Combobox(
            cadre,
            state="readonly",
            values=[
                "Équation normale",
                "QR",
                "SVD",
                "Descente de gradient",
                "Newton",
                "BFGS",
            ],
        )
        self.choix_methode.current(1)
        self.choix_methode.grid(row=7, column=1, sticky="ew", pady=5)

        ttk.Label(cadre, text="Pas du gradient (si besoin) :").grid(row=8, column=0, sticky="w", pady=5)
        self.entree_pas = ttk.Entry(cadre)
        self.entree_pas.insert(0, "0.01")
        self.entree_pas.grid(row=8, column=1, sticky="ew", pady=5)

        # Boutons
        cadre_boutons = ttk.Frame(cadre)
        cadre_boutons.grid(row=9, column=0, columnspan=2, pady=20)

        bouton_lancer = ttk.Button(
            cadre_boutons,
            text="Lancer l’optimisation",
            command=self.lancer_optimisation,
        )
        bouton_lancer.grid(row=0, column=0, padx=5)

        bouton_afficher = ttk.Button(
            cadre_boutons,
            text="Afficher le profil",
            command=self.afficher_resultat,
        )
        bouton_afficher.grid(row=0, column=1, padx=5)

        bouton_convergence = ttk.Button(
            cadre_boutons,
            text="Afficher la convergence",
            command=self.afficher_convergence_resultat,
        )
        bouton_convergence.grid(row=0, column=2, padx=5)

        bouton_quitter = ttk.Button(
            cadre_boutons,
            text="Quitter",
            command=self.fenetre.destroy,
        )
        bouton_quitter.grid(row=0, column=3, padx=5)

        # Zone de résultats
        ttk.Label(cadre, text="Résultats :").grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 5))

        self.zone_resultats = tk.Text(cadre, height=10, width=70)
        self.zone_resultats.grid(row=11, column=0, columnspan=2, sticky="nsew")

        cadre.columnconfigure(1, weight=1)
        cadre.rowconfigure(11, weight=1)

    # fonction qui lit et convertit les paramètres saisie par l'utilisateur
    def lire_parametres(self):
        try:
            m = float(self.entree_m.get())
            p = float(self.entree_p.get())
            t = float(self.entree_t.get())
            nb_points = int(self.entree_nb_points.get())
            nb_points_controle = int(self.entree_nb_ctrl.get())
            degre = int(self.entree_degre.get())
            pas = float(self.entree_pas.get())
        except ValueError:
            raise ValueError("Un ou plusieurs paramètres sont invalides.")

        return m, p, t, nb_points, nb_points_controle, degre, pas
    
    # fonction qui lance l'optimisation du profil NACA
    def lancer_optimisation(self):
        try:
            m, p, t, nb_points, nb_points_controle, degre, pas = self.lire_parametres()

            self.points_extrados, self.points_intrados = generer_profil_naca(
                m, p, t, nb_points=nb_points
            )

            methode = self.choix_methode.get()

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
                raise ValueError("Méthode inconnue.")

            erreur_totale = (
                self.resultat_extrados["erreur_moyenne"]
                + self.resultat_intrados["erreur_moyenne"]
            )

            self.zone_resultats.delete("1.0", tk.END)
            self.zone_resultats.insert(tk.END, f"Méthode choisie : {self.resultat_extrados['nom_methode']}\n")
            self.zone_resultats.insert(tk.END, f"Erreur moyenne extrados : {self.resultat_extrados['erreur_moyenne']:.6e}\n")
            self.zone_resultats.insert(tk.END, f"Erreur moyenne intrados : {self.resultat_intrados['erreur_moyenne']:.6e}\n")
            self.zone_resultats.insert(tk.END, f"Erreur totale : {erreur_totale:.6e}\n")

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

            messagebox.showinfo("Succès", "Optimisation terminée.")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # fonction (bouton) qui permet d'afficher graphiquement le profil optimiser
    def afficher_resultat(self):
        if self.resultat_extrados is None or self.resultat_intrados is None:
            messagebox.showwarning("Attention", "Lance d’abord une optimisation.")
            return

        afficher_profil(
            self.points_extrados,
            self.points_intrados,
            self.resultat_extrados,
            self.resultat_intrados,
        )

    # fonction (bouton) qui permet d'afficher graphiquement les courbes de convergence
    def afficher_convergence_resultat(self):
        if self.resultat_extrados is None or self.resultat_intrados is None:
            messagebox.showwarning("Attention", "Lance d’abord une optimisation.")
            return

        if "historique_cout_y" not in self.resultat_extrados:
            messagebox.showinfo(
                "Information",
                "Pas de courbe de convergence disponible pour cette méthode directe."
            )
            return

        afficher_convergence(self.resultat_extrados, self.resultat_intrados)

# point d'entrée du programme
if __name__ == "__main__":
    racine = tk.Tk()
    app = ApplicationOptimisation(racine)
    racine.mainloop()