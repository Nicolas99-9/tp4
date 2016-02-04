import numpy as np
import collections
from pprint import pprint

phrase = "aaaabbcd"



def merges(l1,l2):
    lise = []
    for item in range(len(l2)):
        lise.append(l2[item]) 
    for item in range(len(l1)):
        lise.append(l1[item]) 
    return lise


def exo1(words):
    count = collections.Counter(words)
    taille = len(words)
    for element in count:
        count[element]  = count[element]/float(taille)
    sorted(count)
    nombre_differents = len(count)
    tab_finale = [[(0.0,-1,[]) for i in range(nombre_differents)] for j in range(nombre_differents)]
    nb=  nombre_differents -1
    counts = 0
    for element in sorted(count):
        tab_finale[counts][0] = (count[element],-1,[element])
        counts +=1
    colonne = 0
   
    while(nb > 0 ):
        tout_en_bas = tab_finale[nombre_differents-1-colonne][colonne][0]
        pas_en_bas = tab_finale[nombre_differents-2-colonne][colonne][0]
        tab_finale[nombre_differents-2-colonne][colonne] = (tab_finale[nombre_differents-2-colonne][colonne][0],0,tab_finale[nombre_differents-2-colonne][colonne][2])
        tab_finale[nombre_differents-1-colonne][colonne] = (tab_finale[nombre_differents-1-colonne][colonne][0],1,tab_finale[nombre_differents-1-colonne][colonne][2])
        mergedlist = merges(tab_finale[nombre_differents-2-colonne][colonne][2] , tab_finale[nombre_differents-1-colonne][colonne][2])
        
        tab_finale[nombre_differents-2-colonne][colonne+1]  = (tout_en_bas+pas_en_bas,1,mergedlist) 
        colonne +=1
        nb = nb -1
        for i in range(nombre_differents-colonne-1):
            tab_finale[i][colonne] = (tab_finale[i][colonne-1][0],-1,tab_finale[i][colonne-1][2])
        #print2(tab_finale)
        sorted(tab_finale[colonne])
    tab_finale[0][nombre_differents-1] = (tab_finale[0][nombre_differents-1][0],0,tab_finale[0][nombre_differents-1][2]) 
    

    
def print2(tableau):
    for tab in tableau:
        print(tab)


exo1(phrase)

