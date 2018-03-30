# -*- coding:utf-8 -*-
#! /usr/bin/env python
import re
import json
import traceback
from mySQLdataexport import *
import time
import os
#from fenci import Fenci
import jieba
from gensim.models import word2vec
import logging
import sys
import gensim
reload(sys)
sys.setdefaultencoding('utf-8')

stopwords = {}.fromkeys([ line.strip() for line in open("./stopwords.txt") ])
#for i in stopwords:
#	if i == "的":
#	    print type(i)
#exit()
#重新建立word2vec训练词库，删除并打开写句柄
word2vecfile = "word2vec/wordall.txt"
if os.path.exists(word2vecfile):
    os.remove(word2vecfile)
fword2vec = file(word2vecfile,"w+")


def dealfile(rootdir,outputfile):
    print rootdir
    listfile = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    #output = file(outputfile,"w+")
    for i in range(0,len(listfile)):
        path = os.path.join(rootdir,listfile[i])
        #path = os.path.join(rootdir,"10.txt")
        if os.path.isfile(path):
            file1 = open(path,'r')
            filetxt = file1.read()
	    fenciword = jieba.cut(filetxt,cut_all=False) 
	    oneline=""
	    #print listfile[i]
	    for k,v in enumerate(fenciword):
		vutf8 = v.encode("utf-8")
	        if vutf8 not in stopwords:
		    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
		    if not zh_pattern.match(v):
			continue
		    else:
		        #print vutf8
			#print type(vutf8)   
			oneline=oneline+vutf8+" "
		else:
		    continue   
		#print v
	   #print oneline 
	    #output.write(oneline.encode("utf-8")+'\n')
	    fword2vec.write(oneline.encode("utf-8")+'\n')
   #output.close()
    return





#C000008
rootdir = 'cnn-text/C000008'
outputfile = "rt-polaritydata/rt-caijing.dat"
dealfile(rootdir,outputfile)
#C000010
rootdir = 'cnn-text/C000010'
outputfile = "rt-polaritydata/rt-it.dat"
dealfile(rootdir,outputfile)
#C000013
rootdir = 'cnn-text/C000013'
outputfile = "rt-polaritydata/rt-healthy.dat"
dealfile(rootdir,outputfile)
#000014
rootdir = 'cnn-text/C000014'
outputfile = "rt-polaritydata/rt-tiyu.dat"
dealfile(rootdir,outputfile)
#000016
rootdir = 'cnn-text/C000016'
outputfile = "rt-polaritydata/rt-lvyou.dat"
dealfile(rootdir,outputfile)
#000020
rootdir = 'cnn-text/C000020'
outputfile = "rt-polaritydata/rt-jiaoyu.dat"
dealfile(rootdir,outputfile)
#000022
rootdir = 'cnn-text/C000022'
outputfile = "rt-polaritydata/rt-zhaopin.dat"
dealfile(rootdir,outputfile)
#000023
rootdir = 'cnn-text/C000023'
outputfile = "rt-polaritydata/rt-wenhua.dat"
dealfile(rootdir,outputfile)
#000024
rootdir = 'cnn-text/C000024'
outputfile = "rt-polaritydata/rt-junshi.dat"
dealfile(rootdir,outputfile)


#word2vec 训练词向量
class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname

     def __iter__(self):
         #for fname in os.listdir(self.dirname):
         for root,dirs,files in os.walk(self.dirname):
            for file_name in files:
                for line in open(os.path.join(root,file_name)):
                    yield line.strip().split()

start=time.clock()
logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s', level=logging.INFO)
mycorpus=MySentences('/data/deeplearning/cnn-text-classification-tf-master/data/word2vec')
model = word2vec.Word2Vec(mycorpus,size=200,workers=1000,sg=0,hs=1,window=6,min_count=5,cbow_mean=1, batch_words=10000)
# 保存模型，以便重用
model.save(u"_tourism.model")
# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format(u"_tourism.model.txt", binary=False)
for k,v in enumerate(model.wv.vocab):
        print v
fword2vec.close()



def dealfile2(rootdir,outputfile):
    print rootdir
    listfile = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    output = file(outputfile,"w+")
    for i in range(0,len(listfile)):
        path = os.path.join(rootdir,listfile[i])
        #path = os.path.join(rootdir,"10.txt")
        if os.path.isfile(path):
            file1 = open(path,'r')
            filetxt = file1.read()
            fenciword = jieba.cut(filetxt,cut_all=False)
            oneline=""
            #print listfile[i]
            for k,v in enumerate(fenciword):
                vutf8 = v.encode("utf-8")
                if vutf8 not in model.wv.vocab:
                    continue
		else:
                    oneline=oneline+vutf8+" "
                #print v
           #print oneline 
            output.write(oneline.encode("utf-8")+'\n')
    output.close()
    return


#C000008
rootdir = 'cnn-text/C000008'
outputfile = "rt-polaritydata/rt-caijing.dat"
dealfile2(rootdir,outputfile)
#C000010
rootdir = 'cnn-text/C000010'
outputfile = "rt-polaritydata/rt-it.dat"
dealfile2(rootdir,outputfile)
#C000013
rootdir = 'cnn-text/C000013'
outputfile = "rt-polaritydata/rt-healthy.dat"
dealfile2(rootdir,outputfile)
#000014
rootdir = 'cnn-text/C000014'
outputfile = "rt-polaritydata/rt-tiyu.dat"
dealfile2(rootdir,outputfile)
#000016
rootdir = 'cnn-text/C000016'
outputfile = "rt-polaritydata/rt-lvyou.dat"
dealfile2(rootdir,outputfile)
#000020
rootdir = 'cnn-text/C000020'
outputfile = "rt-polaritydata/rt-jiaoyu.dat"
dealfile2(rootdir,outputfile)
#000022
rootdir = 'cnn-text/C000022'
outputfile = "rt-polaritydata/rt-zhaopin.dat"
dealfile2(rootdir,outputfile)
#000023
rootdir = 'cnn-text/C000023'
outputfile = "rt-polaritydata/rt-wenhua.dat"
dealfile2(rootdir,outputfile)
#000024
rootdir = 'cnn-text/C000024'
outputfile = "rt-polaritydata/rt-junshi.dat"
dealfile2(rootdir,outputfile)



