import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(caijing_data_file,it_data_file,healthy_data_file,tiyu_data_file,lvyou_data_file,jiaoyu_data_file,zhaopin_data_file,wenhua_data_file,junshi_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    caijing_examples = list(open(caijing_data_file, "r").readlines())
    caijing_examples = [s.strip() for s in caijing_examples]
#    for s in positive_examples:
#	print s
    it_examples = list(open(it_data_file, "r").readlines())
    it_examples = [s.strip() for s in it_examples]
    healthy_examples = list(open(healthy_data_file, "r").readlines())
    healthy_examples = [s.strip() for s in healthy_examples]
    tiyu_examples = list(open(tiyu_data_file, "r").readlines())
    tiyu_examples = [s.strip() for s in tiyu_examples]
    lvyou_examples = list(open(lvyou_data_file, "r").readlines())
    lvyou_examples = [s.strip() for s in lvyou_examples]
    jiaoyu_examples = list(open(jiaoyu_data_file, "r").readlines())
    jiaoyu_examples = [s.strip() for s in jiaoyu_examples]
    zhaopin_examples = list(open(zhaopin_data_file, "r").readlines())
    zhaopin_examples = [s.strip() for s in zhaopin_examples]
    wenhua_examples = list(open(wenhua_data_file, "r").readlines())
    wenhua_examples = [s.strip() for s in wenhua_examples]
    junshi_examples = list(open(junshi_data_file, "r").readlines())
    junshi_examples = [s.strip() for s in junshi_examples]
    # Split by words
    x_text = caijing_examples + it_examples + healthy_examples + tiyu_examples + lvyou_examples + jiaoyu_examples + zhaopin_examples + wenhua_examples + junshi_examples
    #print "x_text length"
    #print len(x_text)
    #for s in x_text: 
    #    print s
    #x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    caijing_labels = [[1,0,0,0,0,0,0,0,0] for _ in caijing_examples]
    it_labels = [[0,1,0,0,0,0,0,0,0] for _ in it_examples]
    healthy_labels = [[0,0,1,0,0,0,0,0,0] for _ in healthy_examples]
    tiyu_labels = [[0,0,0,1,0,0,0,0,0] for _ in tiyu_examples]
    lvyou_labels = [[0,0,0,0,1,0,0,0,0] for _ in lvyou_examples]
    jiaoyu_labels = [[0,0,0,0,0,1,0,0,0] for _ in jiaoyu_examples]
    zhaopin_labels = [[0,0,0,0,0,0,1,0,0] for _ in zhaopin_examples]
    wenhua_labels = [[0,0,0,0,0,0,0,1,0] for _ in wenhua_examples]
    junshi_labels = [[0,0,0,0,0,0,0,0,1] for _ in junshi_examples]
    y = np.concatenate([caijing_labels,it_labels,healthy_labels,tiyu_labels,lvyou_labels,jiaoyu_labels,zhaopin_labels,wenhua_labels,junshi_labels], 0)
    #print "x_text length"
    #print len(x_text)
    #for s in x_text: 
    #    print s
    
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    #data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
