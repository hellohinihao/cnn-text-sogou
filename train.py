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
tf.flags.DEFINE_string("caijing_data_file", "./data/rt-polaritydata/rt-caijing.dat", "Data source for the caijing data.")
#C000010
tf.flags.DEFINE_string("it_data_file", "./data/rt-polaritydata/rt-it.dat", "Data source for the it data.")
#C000013
tf.flags.DEFINE_string("healthy_data_file", "./data/rt-polaritydata/rt-healthy.dat", "Data source for the healthy data.")
#000014
tf.flags.DEFINE_string("tiyu_data_file", "./data/rt-polaritydata/rt-tiyu.dat", "Data source for the tiyu data.")
#000016
tf.flags.DEFINE_string("lvyou_data_file", "./data/rt-polaritydata/rt-lvyou.dat", "Data source for the lvyou data.")
#000020
tf.flags.DEFINE_string("jiaoyu_data_file", "./data/rt-polaritydata/rt-jiaoyu.dat", "Data source for the jiaoyu data.")
#000022
tf.flags.DEFINE_string("zhaopin_data_file", "./data/rt-polaritydata/rt-zhaopin.dat", "Data source for the zhaopin data.")
#000023
tf.flags.DEFINE_string("wenhua_data_file", "./data/rt-polaritydata/rt-wenhua.dat", "Data source for the wenhua data.")
#000024
tf.flags.DEFINE_string("junshi_data_file", "./data/rt-polaritydata/rt-junshi.dat", "Data source for the junshi data.")

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
tf.flags.DEFINE_string("word2vec_file", "./data/_tourism.model.txt", "Data source for the other data.")

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
sum = 0
for x in x_text:
	if len(x.split(" "))>2000:
	    sum = sum + 1
print sum
max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 1500
embedding = np.asarray(embd)
xtemp = []
for x in x_text:
	xtemp.append(x.split(" "))
print len(x_text)
print max_document_length
#x= np.zeros([len(x_text),max_document_length])
##a=0
#for index,value in enumerate(xtemp):
#	for k,v in enumerate(value):
#		#a = a+1
#		if v.strip():
#			#print v
#		#print vocabindex[v]
#			x[index][k] = vocabindex[v]	
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
x_shuffled = []
for i in range(shuffle_indices.size):
	x_shuffled.append(x_text[shuffle_indices[i]])
#print len(x_shuffled)
#print type(x_shuffled)
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
	    x= np.zeros([len(x_batch),max_document_length])
	    for index,value in enumerate(x_batch):
                for k,v in enumerate(value):
                    if v.strip():
                       #print v
               #print vocabindex[v]
		        if vocabindex.has_key(v):
                            x[index][k] = vocabindex[v]
            feed_dict = {
              cnn.input_x: x,
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
            x= np.zeros([len(x_batch),max_document_length])
            for index,value in enumerate(x_batch):
                for k,v in enumerate(value):
                    if v.strip():
                       #print v
               #print vocabindex[v]
                        if vocabindex.has_key(v):
                            x[index][k] = vocabindex[v]
            feed_dict = {
              cnn.input_x: x,
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
