import numpy as np
import collections
from pprint import pprint
import operator


phrase = "aaaabbcd"


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return "%s_%s" % (self.left, self.right)

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
    count = sorted(count.items(), key=operator.itemgetter(1),reverse = True)
    nombre_differents = len(count)
    tab_finale = [[(0.0,-1,[]) for i in range(nombre_differents)] for j in range(nombre_differents)]
    nb=  nombre_differents -1
    counts = 0
    for (element,b) in count:
        print(element)
        tab_finale[counts][0] = (b,-1,[element])
        counts +=1
    for j in range(0,nombre_differents):
        for i in range(1,nombre_differents):
            tab_finale[j][i] = (tab_finale[j][i-1][0],-1,tab_finale[j][i-1][2])
    colonne = 0
    #print2(tab_finale)
    while(nb > 0 ): 
        
        tout_en_bas = tab_finale[nombre_differents-1-colonne][colonne][0]
        pas_en_bas = tab_finale[nombre_differents-2-colonne][colonne][0]
        tab_finale[nombre_differents-2-colonne][colonne] = (tab_finale[nombre_differents-2-colonne][colonne][0],0,tab_finale[nombre_differents-2-colonne][colonne][2])
        tab_finale[nombre_differents-1-colonne][colonne] = (tab_finale[nombre_differents-1-colonne][colonne][0],1,tab_finale[nombre_differents-1-colonne][colonne][2])
        mergedlist = merges(tab_finale[nombre_differents-2-colonne][colonne][2] , tab_finale[nombre_differents-1-colonne][colonne][2])
        
        tab_finale[nombre_differents-2-colonne][colonne+1]  = (tout_en_bas+pas_en_bas,-1,mergedlist) 
        colonne +=1
        nb = nb -1
        
        for i in range(nombre_differents-colonne-1):
            tab_finale[i][colonne] = (tab_finale[i][colonne-1][0],-1,tab_finale[i][colonne-1][2])
        
        sortes(tab_finale,colonne)
       
    tab_finale[0][nombre_differents-1] = (tab_finale[0][nombre_differents-1][0],0,tab_finale[0][nombre_differents-1][2]) 
    #print3(tab_finale)
    finale = {}
    finished = {}
    for element,v in count:
        
        finale[element] = "" 
    nb=  nombre_differents -2
    while(nb>=0):
        current = []
        for i in range(nombre_differents):
            current.append(tab_finale[i][nb])
        print("------------------------------,\n",current)
        for _,incre,arrive in current:
            for element in finale:
                if(element in arrive and (incre != -1)):
                    finale[element] =str(incre)+finale[element] 
        nb -= 1
    print(finale)
    

def print3(tableau):
    for tab in tableau:
        em = ""
        for _,_,a in tab:
            em = em + ''.join(a)
        print(a)


    
def print2(tableau):
    for tab in tableau:
        print(tab)

def sortes(tableau,colonne):
    liste_temp = []
    for i in range(len(tableau)):
        liste_temp.append(tableau[i][colonne])
    sorted_by_second = sorted(liste_temp, key=lambda tup: tup[0],reverse = True)
    for i in range(len(sorted_by_second)):
        tableau[i][colonne] = sorted_by_second[i]

#----------------------------------------------------------------------------------------
string  = ""
for line in open("english.txt", "r"):
        line = line.strip().lower()
        for ch in line:
            if(ch.isalpha()):
                string += ch
        



def huffmanCodeTree(node, left=True, binString=""):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffmanCodeTree(l, True, binString + "0"))
    d.update(huffmanCodeTree(r, False, binString + "1"))
    return d

freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

#Sort the frequency table based on occurrence this will also convert the
#dict to a list of tuples
freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

while len(nodes) > 1:
    key1, c1 = nodes[-1]
    key2, c2 = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    # Re-sort the list
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)


huffmanCode = huffmanCodeTree(nodes[0][0])  


code = {}
print " Char | Freq  | Huffman code "
print "-----------------------------"
for char, frequency in freq:
    print " %-4r | %5d | %12s" % (char, frequency, huffmanCode[char])
    code[char] = huffmanCode[char]
#exo2("AAAAABBBCCCCDDE".lower())


def encode(mess,code):
    #print([code[mess[s]] for s in range(len(mess))])
    return ''.join([code[mess[s]] for s in range(len(mess))])

encode(string,code)

def taux_compression():
    print("TAUX de compression : ", (1-(len(encode(string,code))/float(len(string)*7)))*100)

taux_compression()

def decode(mot):
    result =""
    bufferss = ""
    print(code)
    for char in range(len(mot)):
        bufferss += mot[char]
        print(bufferss)
        if(bufferss in code.values()):
            result = result + code.keys()[code.values().index(bufferss)]
            bufferss = ""
    print("Mots decode : ", result)


def calculate_entropy():
    count = collections.Counter(string)
    taille = len(string)
    for element in count:
        count[element]  = count[element]/float(taille)
    result = 0.0
    for i in count:
        result += count[i]* np.log2((1.0/count[i]))
    print("entropie : ",result)
calculate_entropy()

#decode(encode(string,code))


