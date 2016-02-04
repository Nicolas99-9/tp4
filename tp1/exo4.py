import numpy as np
import codecs
from string import ascii_letters
import operator
def nb(auth):
	dico = {}
	for s in auth:
		if(dico.has_key(s)):
			dico[s] += 1
		else:
			dico[s] = 1
	for s in dico:
		buffer ="book"
		if(dico[s])>1:
			buffer += "s"
		#print "{0} wreote".format(buffer)  => {0} indice du format
		print s ," wrote ", dico[s] , buffer


def moyenne(msls):
	res = 0
	for s in msls:
		res += s
	return res/float(len(msls))
	

def variance(list):
	return np.var(list)

def sortes(ma_list):
	print sorted(ma_list.items(), key=operator.itemgetter(1))

def open(file):
	count = 0
	diff = []
	my_set = []
	my_charact = {}
	with codecs.open(file,"rt",encoding="utf8") as my_file:
		for line in my_file:
			line= line.strip() # remove the \n
			mots = line.split(" ")
			my_set.append("")
			for s in line:
				if(s in ascii_letters):
					if(my_charact .has_key(s)):
						my_charact [s] += 1
					else:
						my_charact [s] = 1
			for s in mots:
				if(not(s in diff)):
					diff.append(s)
				if(len(my_set[count])<len(s)):
					my_set[count] = s
			count += 1
	print "mots differents" , len(diff)
	for s in range(len(my_set)):
		print "Number de ligne {0} et mots : {1}".format(s,my_set[s])
	print count
	sortes(my_charact)
	

lst = [1,5,8,7,2]

print "moyenne : " , moyenne(lst)
print "variance :" , variance(lst)

print {'ezlke' :12,'klkl' : 25}
authors = ["S", "C" , "B","S","S"]
nb(authors)
open("english.txt")
