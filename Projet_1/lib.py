# Ben Kabongo B.
# 25 Jan 2022
#
# IA et Jeux LU3IN025
# Projet 1


import numpy as np
import time


# Exercice 1

# Question 1

def readPrefEtu() -> tuple:
    """
    lecture des préférences des étudiants
    :return (np.ndarray, list):
        - la matrice de préférences des étudiants sur les spécialités
        - les noms des étudiants
    """
    lines = open("PrefEtu.txt").readlines()
    # nombre d'étudiants
    nb = int(lines[0])
    # noms des étudiants
    names = []
    # matrice de préférences
    prefs = np.zeros((nb, 9), dtype=np.int)
    for i in range(1, nb + 1):
        line = lines[i].split("\t")
        names.append(line[1])
        prefs[i-1] = list(map(int, line[2:]))
    return prefs, names


def readPrefSpe():
    """
    lecture des préférences des spécialités
    :return
        - np.ndarray: matrice des préférences
        - list: liste des noms des spécialités
        - list: liste des capacités des spécialités
    """
    lines = open("PrefSpe.txt").readlines()
    # nombre d'étudiants
    nb = int(lines[0].split(" ")[1])
    # noms des spécialités
    names = []
    # capacités des spécialités
    capacities = list(map(int, lines[1].split(" ")[1:]))
    # préférences des spécialités
    prefs = np.zeros((9, nb), dtype=np.int)
    for i in range(2, 11):
        line = lines[i].split("\t")
        names.append(line[1])
        prefs[i-2] = list(map(int, line[2:]))
    return prefs, names, capacities


# Question 2

def etuGaleShapley(prefEtu: np.ndarray, prefSpe: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """
    Gale-Shapley côté étudiant
    :param prefEtu: préférences des étudiants
    :param prefSpe: préférences des spécialités
    :param capacities: capacités des spécialités
    :return
    tableau 2D : première colonne : les numéros des spécialités
                  seconde colonne : un étudiant affecté à la spécialité
    """
    nb = len(prefEtu)
    # 3e colonne : index de l'étudiant dans la préférence de sa spécialité courante
    couples = np.ones((nb, 3), dtype=np.int) * -1
    couples[:, 0] = range(nb)
    propositions = np.zeros(nb, dtype=np.int) # index des prochaines propositions par étudiants
    counters = np.zeros(9, dtype=np.int)      # nombre d'étudiants couramment admis par spécialité

    where = np.where(couples[:, 1] == -1)[0]
    while len(where) != 0:
        e = np.min(where)                       # numéro de l'étudiant
        s = prefEtu[e, propositions[e]]         # proposition de e
        counters[s] += 1
        couples[e, 1] = s
        couples[e, 2] = np.where(prefSpe[s] == e)[0][0]
        # vérification de possibilité d'affectation
        if counters[s] > capacities[s]:
            counters[s] = capacities[s]
            arr = couples[np.where(couples[:, 1] == s)] # liste des étudiants dont la spécialité est s
            arr = arr[arr[:, 2].argsort()]              # par ordre de préférence de s
            i = int(arr[-1, 0]) # dernier étudiant
            # ssi e n'est dernier dans la liste, s l'admet
            if i != e:
                couples[i, [1, 2]] = -1 # retrait de i
            else:
                couples[e, [1, 2]] = -1 # retrait de e
        propositions[e] += 1
        where = np.where(couples[:,1]==-1)[0]
    return couples[:, [0, 1]]


# question 3

def speGaleShapley(prefSpe: np.ndarray, prefEtu: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """
    Gale-Shapley côté spécialité
    :param prefSpe: préférences des spécialités
    :param prefEtu: préférences des étudiants
    :param capacities: capacités des spécialités
    :return
    tableau 2D : première colonne : les numéros des étudiants
                  seconde colonne : leurs spécialités
    """
    nb = len(prefEtu)
    couples = np.ones((nb, 2), dtype=np.int) * -1
    couples[:, 0] = np.concatenate([np.repeat(i, capacities[i]) for i in range(9)])
    propositions = np.zeros(nb, dtype=np.int)     # index des prochaines propositions
    where = np.where(couples[:, 1] == -1)[0]
    while len(where) != 0:
        i = np.min(where)
        s = couples[i, 0]               # index de la spécialité
        e = prefSpe[s, propositions[s]] # proposition de s
        # si e n'a pas encore été affecté
        where_e = np.where(couples[:, 1] == e)[0]
        if len(where_e) == 0:
            couples[i, 1] = e
        else:
            j = where_e[0]
            s_ = couples[j, 0] # index de la spécialité courante
            # changement que si e préfère s à s_
            if np.where(prefEtu[e] == s)[0][0] < np.where(prefEtu[e] == s_)[0][0]:
                couples[i, 1] = e
                couples[j, 1] = -1
        propositions[s] += 1
        where = np.where(couples[:, 1] == -1)[0]
    return couples


# fonctions utiles

def permuteCouples(couples: np.ndarray) -> np.ndarray:
    """
    Change l'ordre des couples (A-B) en (B-A)
    :return la nouvelle disposition
    """
    new = couples.copy()
    new[:, [0, 1]] = new[:, [1, 0]]
    return new


def getNamedCouples(couples: np.ndarray, names1: list, names2: list) -> np.ndarray:
    """
    :param couples: couples tels que formés dans les etu et spe GaleShapley
    :param names1: noms de ceux qui ont proposés
    :param names2: noms de ceux qui ont reçu des propositions
    :return liste des couples nommés
    """
    return np.array([[names1[c[0]], names2[c[1]]] for c in couples])


def preferTo(prefH: np.ndarray, f1, f2):
    """
    :param prefH: liste des préférences de h
    :param f1: un choix de h
    :param f2: un choix de h
    :return true si h préfère f1 à f2
    """
    return np.where(prefH == f1)[0][0] < np.where(prefH == f2)[0][0]


# question 4

def unstablePairs(couples: np.ndarray, pref1: np.ndarray, pref2:np.ndarray) -> np.ndarray:
    """
    :param couples: couples tels que formés dans les etu et spe GaleShapley
    :param names1: noms de ceux qui ont proposés
    :param names2: noms de ceux qui ont reçu des propositions
    :return liste des paires instables
    """
    unstables = []
    n = len(couples)
    for i in range(n):
        for j in range(i + 1, n):
            A, b = couples[i]
            C, d = couples[j]
            # test A - b
            if preferTo(pref1[A], d, b) and preferTo(pref2[d], A, C):
                unstables.append([A, d])
            # test C - b
            if preferTo(pref1[C], b, d) and preferTo(pref2[b], C, A):
                unstables.append([C, b])
    return np.array(unstables)


# question 5

def generatePrefEtu(n: int) -> np.ndarray:
    """
    Génère une liste des préférences de n étudiants pour les 9 spécialités
    :param n: nombre d'étudiants
    :return liste des préférences des étudiants pour les spécialités
    """
    a = np.repeat([range(9)], n, axis=0)
    list(map(np.random.shuffle, a))
    return a


def generatePrefSpe(n: int) -> np.ndarray:
    """
    Génère une liste des préférences de 9 spécialités pour les n étudiants
    :param n: nombre d'étudiants
    :return liste des préférences des spécialités pour les étudiants
    """
    a = np.repeat([range(n)], 9, axis=0)
    list(map(np.random.shuffle, a))
    return a


def generateCapacities(n: int) -> np.ndarray:
    """
    Génère des capacités d'accueils pour les 9 spécialités
    :return liste des capacités
    """
    return 1 + np.random.multinomial(n-9, np.ones(9)/9, size=1)[0]


# question 6

def getTimesStats(start=200, end=2000, step=200, repeat=10):
    """
    Effectue des statistiques sur le temps d'exécution
    :param start: taille de l'instance de départ
    :param end: taille de l'instance d'arrêt
    :param step: pas par taille d'instance
    :param repeat: nombre de repétition par taille d'instance
    :return
        - np.ndarray: tableau des différentes tailles d'instances
        - np.ndarray: tableau des temps moyens pour Gale Shapley côté étudiant
        - np.ndarray: tableau des temps moyens pour Gale Shapley côté spécialité
    """
    N = np.arange(start, end + 1, step)
    TEtu = np.zeros_like(N)
    TSpe = np.zeros_like(N)
    for i in range(len(N)):
        n = N[i]
        te = 0
        ts = 0
        for j in range(repeat):
            prefEtu = generatePrefEtu(n)
            prefSpe = generatePrefSpe(n)
            capacities = generateCapacities(n)
            t = time.time()
            _ = etuGaleShapley(prefEtu, prefSpe, capacities)
            te += (time.time() - t) * 10**3
            t = time.time()
            _ = speGaleShapley(prefSpe, prefEtu, capacities)
            ts += (time.time() - t) * 10**3
        te /= repeat
        ts /= repeat
        TEtu[i] = te
        TSpe[i] = ts
    return N, TEtu, TSpe

# question 7


def main():
    prefEtu, namesEtu = readPrefEtu()
    print("Etudiants :")
    print(namesEtu)
    print(prefEtu)
    print()

    prefSpe, namesSpe, capacities = readPrefSpe()
    print("Spécialités")
    print(namesSpe)
    print("Capacités :", capacities)
    print(prefSpe)
    print()

    etuCouples = etuGaleShapley(prefEtu, prefSpe, capacities)
    print("Etudiants Gale Shapley :")
    print(etuCouples)
    print(getNamedCouples(etuCouples, namesEtu, namesSpe))
    print("Paires instables :")
    print(unstablePairs(etuCouples, prefEtu, prefSpe))
    print()

    speCouples = speGaleShapley(prefSpe, prefEtu, capacities)
    print("Spécialités Gale Shapley :")
    print(speCouples)
    print(getNamedCouples(speCouples, namesSpe, namesEtu))
    print("Paires instables :")
    print(unstablePairs(speCouples, prefSpe, prefEtu))
    print()


if __name__ == '__main__':
    main()
