# 实验参考
本实验借鉴了
<p>https://www.csdn.net/article/2015-11-11/2826192</p>
学习了卷积神经网络在自然语言处理中的应用。
</p>github地址https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py</p>
使用上述github地址cnn-text代码进行修改完成实验目的。
本实验使用了word2vec开源库进行特征提取。

# 实验背景
 <p>文本挖掘是一门交叉性学科,涉及数据挖掘、机器学习、模式识别、人工智能、统计学、计算机语言学、计算机网络技术、信息学等多个领域。文本挖掘就是从大量的文档中发现隐含知识和模式的一种方法和工具,它从数据挖掘发展而来,但与传统的数据挖掘又有许多不同。文本挖掘的对象是海量、异构、分布的文档(web);文档内容是人类所使用的自然语言,缺乏计算机可理解的语义。传统数据挖掘所处理的数据是结构化的,而文档(web)都是半结构或无结构的。所以,文本挖掘面临的首要问题是如何在计算机中合理地表示文本,使之既要包含足够的信息以反映文本的特征,又不至于过于复杂使学习算法无法处理。</p>
 <p>具体到本实验，通过对已打标签搜狗新闻库文本进行特征提取，抽象出特征向量来表达文本信息，将语料从一个无结构的原始文本转化为结构化的计算机可以识别处理的信息，进而将该信息使用卷积神经网络进行训练，以获得可以对未知文本科学分类的模型。该模型可以对更大数据量的未知文本快速分类，使用户对信息的获取更加具有针对性，同时这种训练方法可以应用到情感分析等场景下，抓取微博，知乎等网络媒体内容监控最新网络舆情。</p>
# 实验目的

<p>本案例以搜狗新闻库为例，研究文本数据挖掘中的文本分类问题，即通过对已打标签的语料数据进行训练，建立分类模型实现对未知数据的准确分类。</p>
<p>了解提取文本特征向量特征的常用方法</p>
<p>了解文本分类的常用算法</p>
<p>了解卷积神经网络在自然语言处理中的应用原理</p>
<p>实践使用jieba，word2vec方法库提取文章特征向量</p>
<p>实践使用TensorFlow建立卷积神经网络文本分类模型</p>

# 预备知识
<p>熟悉Linux系统，学习Linux的基本操作指令</p>
<p>python基础知识，了解Tensorflow，word2vec基本使用原理</p>
<p>了解文本分类基础知识</p>

# 实验环境
<p>1.硬件环境：双核、4G内存、30G硬盘</p>
<p>2.操作系统：Ubuntu16.04</p>
<p>3.基础环境：python运行环境，基础python库，gcc编译器等</p>
# 实验步骤
<p>一. 了解文本数据挖掘在日常生活中的应用</p>
<p>文本数据挖掘作为数据挖掘的一个新主题 引起了人们的极大兴趣，同时它也是一个富于争议的研究方向。目前其定义尚无统一的结论，需要国内外学者开展更多的研究以进行精确的定义，类似于我们熟知的数据挖掘定义。我们对文本挖掘作如下定义。</p>
<p></p>
<p>文本挖掘是指从大量文本数据中抽取事先未知的可理解的最终可用的信息或知识的过程。直观地说，当数据挖掘的对象完全由文本这种数据类型组成时，这个过程就称为文本挖掘。</p>
<p>存储信息使用最多的是文本，所以文本挖掘被认为比数据挖掘具有更高的商业潜力，当数据挖掘的对象完全由文本这种数据类型组成时，这个过程就称为文本数据挖掘，事实上，最近研究表明公司信息有80%包含在文本文档中，文本数据挖掘可以细分为以下几种形式：</p>
<p>
1.文本分类<br>文本分类指按照预先定义的主题类别，为文档集合中的每个文档确定一个类别。这样用户不但能够方便地浏览文档，而且可以通过限制搜索范围来使文档的查找更容易、快捷。目前，用于英文文本分类的分类方法较多，用于中文文本分类的方法较少，主要有朴素贝叶斯分类（Naïve Bayes），向量空间模型（Vector Space Model）以及线性最小二乘LLSF(Linear Least Square Fit)。</p>
<p>2.文本聚类<br>聚类与分类的不同之处在于，聚类没有预先定义好的主体类别，它的目标是将文档集合分成若干个簇，要求同一簇内文档内容的相似度尽可能的大，而不同簇之间的相似度尽可能的小。</p>
<p>3.文本结构分析<br>其目的是为了更好地理解文本的主题思想，了解文本表达的内容以及采用的方式，最终结果是建立文本的逻辑结构，即文本结构树，根结点是文本主题，依次为层次和段落。</p>
<p>根据以上各种文本数据挖掘的形式，它在人们的生活中应用也非常的广泛，例如文本数据挖掘可以应用于推荐系统中，今日头条，一点咨询等基于推荐的应用已经成为了人们日常生活中获取信息的主要来源，文本挖掘在推荐系统中的价值在于特征词权重的计算。比如我们给用户推荐一本新书。我们可以按照下面的方式进行建模：首先找到用户评论中关于书籍的所有特征词汇，建立特征词典；然后通过文本分析和时间序列分析结合用户评论的内容和时间紧凑度计算特征词的权重，表示某个用户关心的某个特征的程度。对建立好的用户评论特征程度表进行倒排索引，找到每个特征词的所有评价用户及其评价的权重，最后根据要推荐的书籍的特征找到可以推荐的用户列表，找到评论权重高的用户并把书籍推荐给他。</p>
<p>同时文本数据挖掘也可以应用于舆情监控系统，可以通过对网络媒体用户行为的收集分析此时用户的情感，情感分析就是用户的态度分析。现在大多数情感分析系统都是对文本进行“正负二项分类的”，即只判断文本是正向还是负向的，有的系统也能做到三分类（中立）。比如，要分析用户对2013年“马航370事件”的态度，只要找到该事件的话题文本，通过台大情感词典等工具判断情感词的极性，然后根据一定规则组合情感词的频度和程度即可判断文本的情感。但这种方法无法判断文本的评价刻面。比如，现在有一百万条“小米手机评价”信息，可以通过上面的方法了解大约有百分之多少的用户对小米手机是不满意的，但却无法知道这些不满意的用户是对小米手机的哪一个方面不满意以及占的比率（比如是外形还是性能）。常用的方法是构建小米手机相关词的种子词典，通过词典找到用户评论的刻面，再构建句法树找到评论该刻面的谓语和修饰副词，通过情感词典量化出情感极性，最后将量化后的评论刻面、修饰词、程度副词带入SVM进行文本分类。不过在这里并不适合使用naive bayes，因为在多刻面多分类中，naive bayes很容易出现过拟合。</p>
<p>除此之外，文本数据挖掘还可以应用于医疗，搜索等等领域，作为一个较新的研究方向，文本数据挖掘具有着广拓的应用前景。</p>
<p>二，了解文本分类中特征提取的常用方式</p>
<p>对文本进行分类首先需要提取文本的相应特征，然后把特征放入相应的算法进行训练从而得到最终模型，文本特征提取的质量直接关系到最终模型的有效性，常用的文本特征提取方式有一下几种。</p>
<p>1、TF-IDF<br>
 单词权重最为有效的实现方法就是TF-IDF, 它是由Salton在1988 年提出的。其中TF（term frequency） 称为词频, 用于计算该词描述文档内容的能力; IDF（inverse document frequency） 称为逆文档频率, 用于计算该词区分文档的能力。TF-IDF 的指导思想建立在这样一条基本假设之上: 在一个文本中出现很多次的单词, 在另一个同类文本中出现次数也会很多, 反之亦然。所以如果特征空间坐标系取TF 词频作为测度, 就可以体现同类文本的特点。
其中分子表示该词在此文档中出现的次数，分母表示此文档所有词的总数。另外还要考虑单词区别不同类别的能力, TF-IDF 法认为一个单词所在的文本出现的频率越小, 它区别不同类别的能力就越大, 所以引入了逆文本频度IDF 的概念。 <br>
其中分子表示文档库总文档数，分母表示包含该单词的文档总数，加1是为了编码分母出现0的情况。<br> 
以TF 和IDF 的乘积作为特征空间坐标系的取值测度。 <br>
其中， Wi表示第i个特征词的权重。用TFIDF算法来计算特征词的权重值是表示当一个词在这篇文档中出现的频率越高，同时在其他文档中出现的次数越少，则表明该词对于表示这篇文档的区分能力越强，所以其权重值就应该越大。将所有词的权值排序, 根据需要可以有两种选择方式:( 1) 选择权值最大的某一固定数n 个关键词;( 2) 选择权值大于某一阈值的关键词。一些实验表示,人工选择关键词, 4∽7 个比较合适, 机选关键词10∽15 通常具有最好的覆盖度和专指度。 <br>
TFIDF算法是建立在这样一个假设之上的：对区别文档最有意义的词语应该是那些在文档中出现频率高，而在整个文档集合的其他文档中出现频率少的词语，所以如果特征空间坐标系取TF词频作为测度，就可以体现同类文本的特点。另外考虑到单词区别不同类别的能力，TFIDF法认为一个单词出现的文本频数越小，它区别不同类别文本的能力就越大。因此引入了逆文本频度IDF的概念，以TF和IDF的乘积作为特征空间坐标系的取值测度，并用它完成对权值TF的调整，调整权值的目的在于突出重要单词，抑制次要单词。但是在本质上IDF是一种试图抑制噪音的加权 ，并且单纯地认为文本频数小的单词就越重要，文本频数大的单词就越无用，显然这并不是完全正确的。IDF的简单结构并不能有效地反映单词的重要程度和特征词的分布情况，使其无法很好地完成对权值调整的功能，所以TFIDF法的精度并不是很高。此外，在TFIDF算法中并没有体现出单词的位置信息，对于Web文档而言，权重的计算方法应该体现出HTML的结构特征。特征词在不同的标记符中对文章内容的反映程度不同，其权重的计算方法也应不同。因此应该对于处于网页不同位置的特征词分别赋予不同的系数，然后乘以特征词的词频，以提高文本表示的效果。</p>
<p>2、词频方法(Word Frequency)：<br>
词频是一个词在文档中出现的次数。通过词频进行特征选择就是将词频小于某一闭值的词删除，从而降低特征空间的维数。这个方法是基于这样一个假设，即出现频率小的词对过滤的影响也较小。但是在信息检索的研究中认为，有时频率小的词含有更多的信息。因此，在特征选择的过程中不宜简单地根据词频大幅度删词。</p>
<p>3、文档频次方法(Document Frequency)：<br>
 文档频数(Document Frequency, DF)是最为简单的一种特征选择算法,它指的是在整个数据集中有多少个文本包含这个单词。在训练文本集中对每个特征计一算它的文档频次，并且根据预先设定的阑值去除那些文档频次特别低和特别高的特征。文档频次通过在训练文档数量中计算线性近似复杂度来衡量巨大的文档集，计算复杂度较低，能够适用于任何语料，因此是特征降维的常用方法。 <br>
在训练文本集中对每个特征计算它的文档频数,若该项的DF 值小于某个阈值则将其删除,若其DF值大于某个阈值也将其去掉。因为他们分别代表了“没有代表性”和“没有区分度”2 种极端的情况。DF 特征选取使稀有词要么不含有用信息,要么太少而不足以对分类产生影响,要么是噪音,所以可以删去。DF 的优点在于计算量很小,而在实际运用中却有很好的效果。缺点是稀有词可能在某一类文本中并不稀有,也可能包含着重要的判断信息,简单舍弃,可能影响分类器的精度。<br> 
文档频数最大的优势就是速度快,它的时间复杂度和文本数量成线性关系,所以非常适合于超大规模文本数据集的特征选择。不仅如此,文档频数还非常地高效,在有监督的特征选择应用中当删除90%单词的时候其性能与信息增益和x2 统计的性能还不相上下。DF 是最简单的特征项选取方法,而且该方法的计算复杂度低, 能够胜任大规模的分类任务。但如果某一稀有词条主要出现在某类训练集中,却能很好地反映类别的特征,而因低于某个设定的阈值而滤除掉,这样就会对分类精度有一定的影响。</p>
<p>4.N—Gram算法<br>
它的基本思想是将文本内容按字节流进行大小为N的滑动窗口操作,形成长度为N的字节片段序列。每个字节片段称为gram,对全部gram的出现频度进行统计,并按照事先设定的阈值进行过滤,形成关键gram列表,即为该文本的特征向量空间,每一种gram则为特征向量维度。由于N—Gram算法可以避免汉语分词的障碍,所以在中文文本处理中具有较高的实用性。中文文本处理大多采用双字节进行分解,称之为bi-gram。但是bigram切分方法在处理20%左右的中文多字词时,往往产生语义和语序方面的偏差。而对于专业研究领域,多字词常常是文本的核心特征,处理错误会导致较大的负面影响。基于N—Gram改进的文本特征提取算法[2],在进行bigram切分时,不仅统计gram的出现频度,而且还统计某个gram与其前邻gram的情况,并将其记录在gram关联矩阵中。对于那些连续出现频率大于事先设定阈值的,就将其合并成为多字特征词。这样通过统计与合并双字特征词,自动产生多字特征词,可以较好地弥补N—Gram算法在处理多字词方面的缺陷。</p>
<p>三，了解文本分类的常用算法</p>
<p>在上一课时中介绍了如何在文本中提取出有效的特征向量,本课时中将介绍对文本进行分类的常用算法。</p>
<p>1.K-最近邻算法<br>  K-最近邻算法是一种基于向量空间模型的类比学习方法。因其简单、稳定、有效的特点，被广泛应用于文本分类中。使用kNN算法分类时，首先将待分类文档通过特征权重计算表示成空间向量形式的特征集合；<br>
然后，根据相应的准则将特征向量与预先确定好类别的样本权重向量进行相关的计算，得到前K个相似度较高的文本；最后，判定该文档的文本类别属性。</p>
<p>2.向量空间距离测度分类算法<br>
该算法的思路十分简单，根据算术平均为每类文本集生成一个代表该类的中心向量，然后在新文本来到时，确定新文本向量,计算该向量与每类中心向量间的距离（相似度），最后判定文本属于与文本距离最近的类。
</p>
<p>3.支持向量机<br>
支持向量机（Support Vector Machine，SVM）最初是由Vapnik提出的，它的基本实现思想是：通过某种事先选择的非线性影射把输入向量x映射到一个高维特征空间Z,在这个空间中构造最优分类超平面。也就是SVM采用输入向量的非线性变换，在特征空间中，在现行决策规则集合上按照正规超平面权值的模构造一个结构，然后选择结构中最好的元素和这个元素中最好的函数，以达到最小化错误率的目标，实现了结构风险最小化原则。</p>
<p>4.决策树分类算法<br>
决策树是被广泛使用的归纳学习方法之一。决策树是用样本的属性作为根节点，用属性的取值作为分支的树结构。它是利用信息论原理对大量样本的属性进行分析和归纳产生的。决策树的根节点是所有样本中信息量最大的属性。树的中间节点是以该节点为根的子树所包含的样本子集中信息量最大的属性。决策树的叶节点是样本的类别值。决策树用于对新样本的分类，即通过决策树对新样本属性值的测试，从树的根节点开始，按照样本属性的取值，逐渐沿着决策树向下，直到树的叶节点，该叶节点表示的类别就是新样本的类别。决策树方法是数据挖掘中非常有效的分类方法，它排除噪音的强壮性以及学习反义表达的能力使其更适合于文本分类。比较著名的决策树算法是ID3算法以及它的后继C4.5、C5等。基本的ID3算法是通过自顶向下构造决策树的。
</p>
<p>5.神经网络算法<br>
它是采用感知算法进行分类，在此种模型中，分类知识被隐式地存储在连接
的权值上，使用迭代算法来确定权值向量，当网络输出判别正确时。权值向量保
持不变，否则进行增加或降低的调整，因此也称奖惩法。一般在神经网络分类法中包括两个部分训练部分和测试部分，以样本的特征项构造输入神经元，特征的数量即为输入神经元的数量，至于隐含层数量和该层神经元的数目要视实际而定。在训练部分通过对相当数量的训练样本的训练得到训练样本输入与输出之间的关系即在不断的迭代调整过程中得到连接权值矩阵。测试部分则是针对用户输入的待测样本的特征得到输出值即该样本的所属的类。<br>
本文采用卷积神经网络的方式实现文本分类，它是神经网络的一种形式。
</p>
<p>四，了解卷积神经网络在文本分类中的应用</p>
详细信息可以阅读文章：https://www.cnblogs.com/yelbosh/p/5808706.html
<p>卷积神经网络最早在计算机视觉中得到了广泛的应用，也是当今绝大多数计算机视觉系统的核心技术。<br>
 Kim Yoon在其论文中提出了用于句子分类的卷积神经网络模型。获得了良好的分类性能，并成为新文本分类架构的标准基准。<br>
 论文中指出所构建的模型如上图所示，第一层网络将词向量嵌入到一个低维的向量中。下一层网络就是利用多个卷积核在前一层网络上进行卷积操作。比如，每次滑动3个，4个或者5个单词。第三层网络是一个max-pool层，从而得到一个长向量，并且添加上 dropout 正则项。最后，我们使用softmax函数对进行分类。<br>
该模型所区别于普通神经网络的在于第二层的卷积层和第三层的池化层，也是卷积神经网络与普通神经网络的最大区别。
普通神经网络把输入层和隐含层进行“全连接(Full Connected)“的设计。从计算的角度来讲，相对较小的文本从整个文本中计算特征是可行的。但是，如果是更大的文本，要通过这种全联通网络的这种方法来学习整个文本上的特征，从计算角度而言，将变得非常耗时。卷积层解决这类问题的一种简单方法是对隐含单元和输入单元间的连接加以限制：每个隐含单元仅仅只能连接输入单元的一部分。例如，每个隐含单元仅仅连接输入文本的一小片相邻区域。由于卷积层的神经元也是三维的，所以也具有深度。卷积层的参数包含一系列过滤器（filter），每个过滤器训练一个深度，有几个过滤器输出单元就具有多少深度。池化（pool）即下采样（downsamples），目的是为了减少特征图。池化操作对每个深度切片独立，相对于卷积层进行卷积运算。通过卷积层和池化层可以减少训练的参数，同时也获取了局部的细节特征。<br>
卷积神经网络具体TensorFlow实现将在最后一节中详细介绍。
</p>
<p></p>

特征提取相关代码：

```python
# -*- coding:utf-8 -*-
#! /usr/bin/env python
import re
import json
import traceback
#from mySQLdataexport import *
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
stopwords = {}.fromkeys([ line.strip() for line in open("/data/deeplearning/cnn/data/stopwords.txt") ])
#for i in stopwords:
#	if i == "的":
#	    print type(i)
#exit()
#重新建立word2vec训练词库，删除并打开写句柄
word2vecfile = "/data/deeplearning/cnn/data/word2vec/wordall.txt"
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


print 123


#C000008
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000008'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-caijing.dat"
dealfile(rootdir,outputfile)
print 123
#C000010
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000010'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-it.dat"
dealfile(rootdir,outputfile)
#C000013
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000013'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-healthy.dat"
dealfile(rootdir,outputfile)
#000014
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000014'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-tiyu.dat"
dealfile(rootdir,outputfile)
#000016
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000016'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-lvyou.dat"
dealfile(rootdir,outputfile)
#000020
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000020'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-jiaoyu.dat"
dealfile(rootdir,outputfile)
#000022
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000022'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-zhaopin.dat"
dealfile(rootdir,outputfile)
#000023
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000023'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-wenhua.dat"
dealfile(rootdir,outputfile)
#000024
rootdir = '/data/deeplearning/cnn/data/cnn-text/C000024'
outputfile = "/data/deeplearning/cnn/data/rt-polaritydata/rt-junshi.dat"
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
mycorpus=MySentences('/data/deeplearning/cnn/data/word2vec')
model = word2vec.Word2Vec(mycorpus,size=200,workers=1000,sg=0,hs=1,window=6,min_count=5,cbow_mean=1, batch_words=10000)
# 保存模型，以便重用
model.save("/data/deeplearning/cnn/data/tourism")
# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format("/data/deeplearning/cnn/data/_tourism.model.txt", binary=False)
for k,v in enumerate(model.wv.vocab):
        print v
fword2vec.close()







```



<p>五，实践Tensorflow建立卷积神经网络搜狗新闻库文本分类模型</p>

训练相关代码：

```python
# -*- coding:utf-8 -*-

#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

#C000008
tf.flags.DEFINE_string("caijing_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-caijing.dat", "Data source for the caijing data.")
#C000010
tf.flags.DEFINE_string("it_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-it.dat", "Data source for the it data.")
#C000013
tf.flags.DEFINE_string("healthy_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-healthy.dat", "Data source for the healthy data.")
#000014
tf.flags.DEFINE_string("tiyu_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-tiyu.dat", "Data source for the tiyu data.")
#000016
tf.flags.DEFINE_string("lvyou_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-lvyou.dat", "Data source for the lvyou data.")
#000020
tf.flags.DEFINE_string("jiaoyu_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-jiaoyu.dat", "Data source for the jiaoyu data.")
#000022
tf.flags.DEFINE_string("zhaopin_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-zhaopin.dat", "Data source for the zhaopin data.")
#000023
tf.flags.DEFINE_string("wenhua_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-wenhua.dat", "Data source for the wenhua data.")
#000024
tf.flags.DEFINE_string("junshi_data_file", "/data/deeplearning/cnn/data/rt-polaritydata/rt-junshi.dat", "Data source for the junshi data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6,7", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#word2vec filename
tf.flags.DEFINE_string("word2vec_file", "/data/deeplearning/cnn/data/_tourism.model.txt", "Data source for the other data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

def loadWord2vec(filename):
    vocab = []
    vocabindex = {}
    embd = []
    word2vecfile = open(filename,'r')
    #first line 
    line = word2vecfile.readline()
    indexid = 0
    for line in word2vecfile.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        vocabindex[row[0]] = indexid
 	indexid=indexid+1
	#print row[1:]
	appendrow = []
	for k,v in enumerate(row[1:]):
		appendrow.append(np.float32(v))
		#print k,type(row[k])
        embd.append(appendrow)
    #print type(embd[1][0])
    #print embd[1]
    print('Loaded Word2vector!')
    word2vecfile.close()
    return vocab,vocabindex,embd



# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.caijing_data_file,FLAGS.it_data_file,FLAGS.healthy_data_file,FLAGS.tiyu_data_file,FLAGS.lvyou_data_file,FLAGS.jiaoyu_data_file,FLAGS.zhaopin_data_file,FLAGS.wenhua_data_file,FLAGS.junshi_data_file)
#vocab 词集合  vocabindex 词对应id  embd 词对应word2vec向量集合
vocab,vocabindex,embd = loadWord2vec(FLAGS.word2vec_file)
max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 1500
embedding = np.asarray(embd)
xtemp = []
for x in x_text:
	xtemp.append(x.split(" "))
print len(x_text)
print max_document_length
x= np.zeros([len(x_text),max_document_length])
for index,value in enumerate(xtemp):
	for k,v in enumerate(value):
		#a = a+1
	        if k>=1500:
		    continue
		if v.strip():
			#print v
		#print vocabindex[v]
			x[index][k] = vocabindex[v]	
#for x in x_text:
#	print len(x.split(" "))
#print max_document_length
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#pretrain = vocab_processor.fit(vocab[0:50]," ")
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Vocabulary Size: {:d}".format(len(vocab)))
#x = np.array(list(vocab_processor.transform(x_text[0])))
#for k in range(1):
#	print x[k]
#	print x_text[k]
#print x
#print np.amax(x)
#print vocab_processor.vocabulary_[0]
#for xx in x:
#	print xx
#x = np.array(list(vocab_processor.fit_transform(x_text)))
#print len(x[1])


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
#print type(shuffle_indices)
#print shuffle_indices
#print type(x_text)
#x_shuffled = []
#for i in range(shuffle_indices.size):
#	x_shuffled.append(x_text[shuffle_indices[i]])
##print len(x_shuffled)
##print type(x_shuffled)
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#print x_train
#print y_train
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Vocabulary Size: {:d}".format(len(vocab)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            #sequence_length=x_train.shape[1],
            sequence_length = max_document_length,
	    num_classes=y_train.shape[1],
            vocab_size=len(vocab),
            #vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
	    word2vecresult=embedding)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
	    #print "train"
	    #print max_document_length
	    #print len(x_batch)
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
	    }
            _, step, summaries, loss, accuracy= sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
	    #print "dev"
            #print max_document_length
            #print len(x_batch)
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
            }
            step, summaries, loss, accuracy= sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs,False)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

```
# 实验结果
<p>对于搜狗语料库中的多分类问题，分类准确率可以达到91%</p>

![result][1]


# 校招算法讨论群
![qun][2]


  [1]: http://47.95.35.43/jielun.png
  [2]: http://47.95.35.43/taolun.png
