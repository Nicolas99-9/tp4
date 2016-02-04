import numpy as np
import codecs
from pprint import pprint
from string import ascii_letters
import operator
import matplotlib.pyplot as plt
import nltk as ns


def sortes(ma_list):
	return sorted(ma_list.items(), key=operator.itemgetter(1),reverse=True)

def opene(file):
        result = {}
        count =0
	with codecs.open(file,"r",encoding="utf8") as my_file:
		for line in my_file:
			line= line.strip() # remove the \n*
			mots = line.split(" ")
			for s in mots:
                                count +=1
                                if(result.has_key(s)):
                                        result[s]+=1
                                else:
                                        result[s]= 1
        for s in result:
                result[s] = result[s]/float(count)
        result = sortes(result)
        return result

def display(vale):
        dic = []
        y = []
        count = 0
        for s in vale:
                y.append(s[1])
                dic.append(count)
                count +=1
        plt.loglog(dic,y,"-")
        plt.ylabel('som numbers')
        plt.show()

vale = opene("newsco.en")
display(vale)
