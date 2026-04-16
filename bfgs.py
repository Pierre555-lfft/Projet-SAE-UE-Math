#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def bfgs(fonction_cout,gradient_fonction,point_initial,pas=1.0,tolerance=1e-8,iterations_max=200):
   

    x = np.array(point_initial, dtype=float)
    n = len(x)

    # Approximation initiale de l'inverse de la Hessienne
    matrice_inverse_hessienne = np.eye(n)

    historique_cout = []
    historique_gradient = []

    for k in range(iterations_max):
        gradient = np.array(gradient_fonction(x), dtype=float)
        norme_gradient = np.linalg.norm(gradient)
        cout = fonction_cout(x)

        historique_cout.append(cout)
        historique_gradient.append(norme_gradient)

        print(
            f"itération {k:3d} | coût = {cout:.6e} | "
            f"||grad|| = {norme_gradient:.6e}"
        )

        if norme_gradient < tolerance:
            print("Arrêt : gradient suffisamment petit.")
            break

        # Direction de descente quasi-Newton
        direction = -matrice_inverse_hessienne @ gradient

        # Mise à jour du point
        nouveau_x = x + pas * direction

        nouveau_gradient = np.array(gradient_fonction(nouveau_x), dtype=float)

        delta = nouveau_x - x
        y = nouveau_gradient - gradient

        produit = float(delta @ y)

        # Condition de courbure pour BFGS
        if produit > 1e-14:
            Sy = matrice_inverse_hessienne @ y
            terme1 = (
                1.0 + (y @ Sy) / produit
            ) * np.outer(delta, delta) / produit
            terme2 = (
                np.outer(delta, Sy) + np.outer(Sy, delta)
            ) / produit

            matrice_inverse_hessienne = (
                matrice_inverse_hessienne + terme1 - terme2
            )
        else:
            print("Avertissement : mise à jour BFGS ignorée (delta^T y trop petit).")

        x = nouveau_x

    return x, historique_cout, historique_gradient

if __name__ == "__main__":

    def fonction_cout(vecteur):
        x, y = vecteur
        return (x - 2.0)**2 + 3.0 * (y + 1.0)**2

    def gradient_fonction(vecteur):
        x, y = vecteur
        return np.array([
            2.0 * (x - 2.0),
            6.0 * (y + 1.0)
        ])

    point_initial = np.array([0.0, 0.0])

    solution, historique_cout, historique_gradient = bfgs(
        fonction_cout=fonction_cout,
        gradient_fonction=gradient_fonction,
        point_initial=point_initial,
        pas=0.25,
        tolerance=1e-10,
        iterations_max=50
    )

    print("\nSolution trouvée :", solution)
    print("Coût final :", fonction_cout(solution))


