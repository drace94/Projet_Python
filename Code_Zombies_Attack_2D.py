# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:16:50 2020

@author: Gabriel Depaillat, Hugo Valayer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import random as rd
import time

# Debut du decompte du temps
start_time = time.time()

def random(a,b):
    """
    Cette fonction retourne un nombre entier dans l'intervalle [a;b]
    """
    return round((b-a)*rd.random() + a)

def Trait_float(nombre,n):
    #met en forme un float tel que 0.12354 devient 0.123, 0.10056 devient 0.1 etc...
    #on garde donc d'abord uniquement les n premières décimales puis on affiche uniquement les chiffres significatifs
    return format(float(format(float(nombre),'.'+str(n)+'f')),'.'+str(n)+'g')

def init_A(alpha,theta,dim):
    """
    Cette fonction sert à initialiser la matrice A. 
    
    D'après Le lemme d'Hadamard, A est une matrice à diagonale strictement dominante alors elle est 
    inversible. A est à diagonale strictement dominante si alpha, theta >= 0
    """
    A = []
    a = -alpha*theta
    b = (1+2*alpha*theta)
    for i in range(dim):
        ligne=[]
        for j in range(dim):
            if i == j :
                ligne.append(b)
            elif j == i-1 or j == i+1:
                ligne.append(a)
            else:
                ligne.append(0)
        A.append(ligne)
    #Conditions aux limites de Von Neumann
    # A[0][0] = A[0][0] + a
    # A[dim-1][dim-1] = A[dim-1][dim-1] + a 
        
    #Conditions aux limites cycliques
    A[0][dim-1] = a
    A[dim-1][0] = a
    return np.array(A)

def init_B(alpha,theta,dim):
    """
    Cette fonction sert à initialiser la matrice B.
    """
    B = []
    a = (1-theta)*alpha
    b = 1-2*alpha*(1-theta)
    for i in range(dim):
        ligne=[]
        for j in range(dim):
            if i == j :
                ligne.append(b)
            elif j == i-1 or j == i+1:
                ligne.append(a)
            else:
                ligne.append(0)
        B.append(ligne)
    # Conditions aux limites de Von Neumann
    # B[0][0] = B[0][0] + a
    # B[dim-1][dim-1] = B[dim-1][dim-1] + a
        
    #Conditions aux limites cycliques
    B[0][dim-1] = a
    B[dim-1][0] = a
    
    return np.array(B)

def init_S(min,max,ligne,colonne):
    M=[]
    ligne_courante=[]
    for i in range(ligne):
        for j in range(colonne):
            ligne_courante.append(random(min,max))
        M.append(ligne_courante)
        ligne_courante=[]
    return np.array(M)

def init_I(ligne,colonne):
    I = np.zeros((ligne,colonne))
    continuer = True
    while continuer :
        Pos_X = eval(input("Position en X ? "))
        Pos_Y = eval(input("Position en Y ? "))
        Val = eval(input("Valeur ? "))
        I[Pos_X][Pos_Y] = Val
        reponse = input("Continuer : oui ou non ? ").lower()
        if reponse != "oui" :
            continuer = False
    return np.array(I)

def diffusion(M,t):
    """
    

    Parameters
    ----------
    M : Matrice de taille M*N sur laquelle on applique l'équation de la chaleur
    t : Temps d'application de l'équation de la chaleur

    Returns
    -------
    Res : La matrice de taille M*N après application l'équation de la chaleur

    """
    Res = M
    lignes=len(M)
    colonnes=len(M[0])
    A1 = init_A(1/4,1/2,lignes)
    B1 = init_B(1/4,1/2,lignes)
    A1_inv = np.linalg.inv(A1)
    A2 = init_A(1/4,1/2,colonnes)
    B2 = init_B(1/4,1/2,colonnes)
    A2_inv = np.linalg.inv(A2)
    for i in range(t):
        #diffusion selon x
        Res = np.dot(A1_inv,np.dot(B1,Res))
        #diffusion selon y
        Res = np.dot(A2_inv,np.dot(B2,np.transpose(Res)))
        Res = np.array(np.transpose(Res))
    return Res
    
# Definition du modele
def Model(x, params):
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    theta = params["theta"]
    pi = params["pi"]
    epsilon = params["epsilon"]

    xdot = np.array([alpha*x[0] - beta*x[0]*x[1] - epsilon*x[0], 
                      beta*x[1]*x[0] - theta*x[1] - gamma*x[1],
                      gamma*x[1] - delta*x[2] - pi*x[2],
                      delta*x[2] - epsilon*x[3],
                      epsilon*x[0] + epsilon*x[3] + theta*x[1] + pi*x[2]])
    return xdot

# Methode RK4
def RK4(f, x0, T, h): # h = pas de temps
    t = np.arange(0, T, h) # vecteur temps
    nt = t.size
    nx = x0.size
    x = np.zeros((nx,nt)) # vecteur solution
    x[:,0] = x0

    for k in range(nt - 1):
        k1 = f(t[k], x[:,k])
        k2 = f(t[k] + h/2, x[:,k] + h*k1/2)
        k3 = f(t[k] + h/2, x[:,k] + h*k2/2)
        k4 = f(t[k] + h, x[:, k] + h*k3)

        x[:,k+1] = x[:,k] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return x

def systeme(x0,T,h):
    params = {"alpha": 0, "beta": 0.003, "gamma": 0.2, "delta": 0.1, "theta": 0.01, "pi": 0.01, "epsilon": 0}
    f = lambda t, x : Model(x, params)
    Res = RK4(f, x0, T, h)
    return Res[:,-1]

def HR_tenseur(tenseur,n):
    for i in range(len(tenseur)):
        for j in range(len(tenseur[0])):
            for k in range(len(tenseur[0][0])):
                tenseur[i,j,k]=Trait_float(tenseur[i,j,k],n)

def main(tenseur,t):
    resultat=[]
    resultat += [tenseur.copy()] # Parce que c'est des pointeurs
    tens = tenseur
    for k in range(t):
        for i in range(len(tenseur[0])):
            for j in range(len(tenseur[0][0])):
                tens[:,i,j] = systeme(tens[:,i,j],1,0.01)
                # tens = np.array(tens)
        tens[0] = diffusion(tens[0],1)
        tens[1] = diffusion(tens[1],1)
        tens[3] = diffusion(tens[3],1)
        resultat += [tens.copy()]         # Parce que c'est des pointeurs
        print("Calcul terminé pour t="+str(k))
    return resultat

# Graphismes
nbr_pts = 10
inter_t =[0,100]
pas_t = 1

# Paramètres & Initialisation des matrices
lignes=100
colonnes=100
S = init_S(100,500,lignes,colonnes)
I = init_I(lignes,colonnes)
D = np.zeros((lignes,colonnes))
P = np.zeros((lignes,colonnes))
M = np.zeros((lignes,colonnes))
tenseur = np.array([S,I,D,P,M]) # de taille lignes*colonnes*5
resultat = main(tenseur,inter_t[1])

def affichage(title,donnees):
    def data_gen(num):
        ax.cla()
        x = np.outer(np.arange(0, lignes, 1), np.ones(lignes))
        y = np.outer(np.arange(0, colonnes, 1),np.ones(colonnes)).T
        z = resultat[num][donnees]
        ax.set_zlim(1, 500)
        surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
        # surf = ax.plot_wireframe(x, y, z, rstride=1, cstride=1,linestyles='dotted')
        ax.set_title("t="+str(num)+" Max="+str(Trait_float(np.amax(z),3))+" Min="+str(Trait_float(np.amin(z),3)))
        return surf
    fig = plt.figure(figsize=(15,10))
    ax = p3.Axes3D(fig)
    ax.set_xlim(0, lignes)
    ax.set_ylim(0, colonnes)
    anim = animation.FuncAnimation(fig,data_gen,np.arange(inter_t[0], inter_t[1], pas_t), interval=200)
    anim.save(title+'.mp4', writer="ffmpeg")

affichage("Population infectée",1)

# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))
