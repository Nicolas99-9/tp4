from string import ascii_letters
from collections import Counter
from pprint import pprint
import numpy as np
from nltk import ngrams
import codecs


def generation2(list):
    result = {}
    result2= {}
    liste = decoupe(list)
    for s in liste:
        if result.has_key(s):
            result[s] = result[s]+ 1
        else:
            result[s] = 1
    for s in result:
        result2[s] = result[s]/float(len(result))
    return result2;

def sum2(list):
    tmp = 0
    for y in list:
        tmp = tmp + list[y]
    return tmp


def generation(list):
    result ={}
    final = {}
    liste = decoupe(list)
    for s in liste:
        a = (s[0])                                                                               
        c = s[1]
        if result.has_key(a):
            if result[a].has_key(s[1]):
                 result[a][c] =  result[a][c] +1
            else:
                result[a][c] = 1  
        else:
            result[a]= {}
            result[a][c] = 1
    for y in result:
        ma_value = sum2(result[y])
        for ee in result[y]:
            result[y][ee]= result[y][ee]/float(ma_value)
    return result
                      
    
'''
def decoupe(l):
    strings = ""
    for e in l:
        strings = strings + e
    print(list(ngrams(strings.split(),2)))
    return 0
'''

def decoupe(l):
    result = []
    taille = len(l)
    for i in range(1,taille):
        result.append((l[i-1],l[i]))     
    return result

def sample_from_discrete_distrib(distrib):
    words, probas = list(zip(*distrib.items()))
    s =  np.random.choice(words, p = probas)
    return s

def getList(file):
        result = {}
        count =0
        motss = []
	with codecs.open(file,"r",encoding="utf8") as my_file:
		for line in my_file:
			line= line.strip() # remove the \n*
			mots = line.split(" ")
			for s in mots:
                                motss.append(s)
                                count +=1
                                if(result.has_key(s)):
                                        result[s]+=1
                                else:
                                        result[s]= 1
        for s in result:
                result[s] = result[s]/float(count)
        return (result,motss)
    
def proba(phrase, probas):
    s = phrase.split()
    count = 1.0
    for e in s:
        if(probas.has_key(e)):
            count *= probas[e]
    return count
        
    
def generationMots(list):
    mon_dico = list
    ma_liste = ["there"]
    while(ma_liste[len(ma_liste)-1] != "."):
          taille = len(ma_liste)
          cle =(ma_liste[taille-1])
          ma_hash = mon_dico[cle]
          eee = sample_from_discrete_distrib(ma_hash)
          ma_liste.append(eee)
    pprint(ma_liste)
    return ma_liste

def decodes(liste):
    buffer = ""
    for i in range(2,len(liste)):
	buffer = buffer + liste[i] + " "
	if(i==2):
	    buffer = buffer.title()
    print buffer

(probas,texte) =  getList("newsco.en")

print(proba("i want that house",probas))
#print(list(decoupe("Stephe is an asshole")))


def summ(en):
    count = 0.0
    for s in en:
        count += en[s]
    return count

def probasBigram(texte,model,uni):
    mots = texte.split()
    red = uni[mots[0]]
    h = mots[0]
    for s in range(1,len(mots)):
        print(mots[s])
        if(model.has_key(h)):
            if(model[h].has_key(mots[s])):
                red *= model[h][mots[s]]
            else:
                model[h][mots[s]] = 1.0/summ(model[h])
                red *= model[h][mots[s]]
        else:
            model[h] ={}
            model[h][mots[s]] = 1.0
            red *= model[h][mots[s]]
        h = mots[s]
    return red
        
    

bigramm = (generation(texte))
print(probasBigram("there is a house mslls msls",bigramm,probas))



print(bigramm["there"]["is"])
print(probas["there"])

generationMots(bigramm)

#decodes(generationMots(listeFinal))
