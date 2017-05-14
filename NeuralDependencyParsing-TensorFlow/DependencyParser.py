import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon
"""

wordDict = {}
posDict = {}
labelDict = {}
system = None


def genDictionaries(sents, trees):
    """
    Generate Dictionaries for word, pos, and arc_label
    Since we will use same embedding array for all three groups,
    each element will have unique ID
    """
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):
    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    pos = []
    label = []
    word = []


    for i in range(2, -1, -1):
        index = c.getStack(i)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))

    for i in range(0, 3, 1):
        index = c.getStack(i)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))

    for i in range(0, 2, 1):
        k = c.getStack(i)
        index = c.getLeftChild(k, 1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k, 1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getLeftChild(k, 2)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k, 2)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getLeftChild(c.getLeftChild(k, 1), 1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(c.getRightChild(k, 1), 1)
        word.append(getWordID(c.getWord(index)))
        pos.append(getPosID(c.getPOS(index)))
        label.append(getLabelID(c.getLabel(index)))

        features = []
        features.extend(word)
        features.extend(pos)
        features.extend(label)

    for i in range(1, len(features) + 1):
        if features[i - 1] == -1:
            features[i - 1] = 0;
    return features;


def genTrainExamples(sents, trees):
    """
    Generate train examples
    Each configuration of dependency parsing will give us one training instance
    Each instance will contains:
        WordID, PosID, LabelID as described in the paper(Total 48 IDs)
        Label for each arc label:
            correct ones as 1,
            appliable ones as 0,
            non-appliable ones as -1
    """
    numTrans = system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    # for i in pbar(list(range(len(sents)))):
    for i in range(len(sents)):
        print(i)
        if trees[i].isProjective():
            c = system.initialConfiguration(sents[i])

            while not system.isTerminal(c):
                oracle = system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                features.append(feat)
                labels.append(label)
                c = system.apply(c, oracle)
    return features, labels


def forward_pass(inputs, weights, biases):
    """
    =======================================================

    Implement the forwrad pass described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =======================================================
    """

    return tf.add(tf.matmul(inputs, tf.transpose(weights)), biases);


if __name__ == '__main__':

    # Load all dataset
    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    # Load pre-trained word embeddings
    dictionary, word_embeds = pickle.load(open('word2vec.model', 'rb'))

    # Create embedding array for word + pos + arc_label
    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print(("Found embeddings: ", foundEmbed, "/", len(knownWords)))

    # Get a new instance of ParsingSystem with arc_labels
    system = ParsingSystem(list(labelDict.keys()))

    print("Generating Training Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    graph = tf.Graph()

    with graph.as_default():

        embeddings = tf.Variable(embedding_array, dtype=tf.float32)

        #hidden_weights = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens * Config.embedding_size],
             #                                            stddev=1.0 / math.sqrt(Config.n_Tokens)))
        #hidden_weights2 = tf.Variable(
            #tf.truncated_normal([Config.hidden_size, Config.batch_size], stddev=1.0 / math.sqrt(Config.n_Tokens)))

        #output_weights = tf.Variable(
            #tf.truncated_normal([system.numTransitions(), Config.hidden_size], stddev=1.0 / math.sqrt(Config.n_Tokens)))

        # Bias matrices
        #hidden_biases = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        #hidden_biases2 = tf.Variable(tf.zeros([Config.hidden_size, 1]))

        #multi = tf.matmul(hidden_weights, tf.transpose(train_input_embeddings))
        #hidden_op1 = tf.pow(tf.add(multi, hidden_biases), 3)

        #activation_hidden = tf.pow(tf.add(tf.matmul(hidden_weights2, tf.transpose(hidden_op1)), hidden_biases2), 3)

        """
        ===================================================================

        Define the computational graph with necessary variables.
        You may need placeholders of:
            train_inputs
            train_labels
            test_inputs

        Implement the loss function described in the paper

        ===================================================================
        """
        t_labels = tf.placeholder(tf.float32, shape=[Config.batch_size, system.numTransitions()])
        t_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])

        tInput_embeddings = tf.nn.embedding_lookup(embeddings, t_inputs)
        tInput_embeddings = tf.reshape(tInput_embeddings,
                                       [Config.batch_size, Config.embedding_size * Config.n_Tokens])



        # Weight matrices
        hidden_weights = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                                         stddev=1.0 / math.sqrt(Config.n_Tokens)))
        output_weights = tf.Variable(
            tf.truncated_normal([system.numTransitions(), Config.hidden_size], stddev=1.0 / math.sqrt(Config.n_Tokens)))

        # Bias matrices
        hidden_biases = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        matrixMultiply = tf.matmul(hidden_weights, tf.transpose(tInput_embeddings))
        activation_hidden = tf.pow(tf.add(matrixMultiply, hidden_biases), 3)
        #activation_hidden = tf.tanh(tf.add(matrixMultiply, hidden_biases))
        prediction = (tf.transpose(tf.matmul(output_weights, activation_hidden)))

        tempLable = tf.arg_max(t_labels, dimension=1)

        # cost function
        crossEntropy_loss = (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=tempLable))

        regularizer = tf.nn.l2_loss(output_weights) + tf.nn.l2_loss(tInput_embeddings) + tf.nn.l2_loss(
            hidden_weights) + tf.nn.l2_loss(hidden_biases)

        # Loss
        loss = tf.reduce_mean(crossEntropy_loss + Config.lam * regularizer)

        # Test data
        test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens])
        test_input_embeddings = tf.nn.embedding_lookup(embeddings, test_inputs)
        test_input_embeddings = tf.reshape(test_input_embeddings, [1, Config.embedding_size * Config.n_Tokens])
        activation_hidden_test = tf.pow(
            tf.add(tf.matmul(hidden_weights, tf.transpose(test_input_embeddings)), hidden_biases), 3)
        test_pred = (tf.transpose(tf.matmul(output_weights, activation_hidden_test)))

        optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)

        # Compute Gradients
        grads = optimizer.compute_gradients(loss)
        # Gradient Clipping
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)

        init = tf.global_variables_initializer()

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:
        init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            print("Step:", step)
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {t_inputs: batch_inputs, t_labels: batch_labels}

            _, loss_val = sess.run([app, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # Display average loss
            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print(("Average loss at step ", step, ": ", average_loss))
                average_loss = 0

            # Print out the performance on dev set
            if step % Config.validation_step == 0 and step != 0:
                print(("\nTesting on dev set at step ", step))
                predTrees = []
                for sent in devSents:
                    numTrans = system.numTransitions()

                    c = system.initialConfiguration(sent)
                    while not system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = system.transitions[j]

                        c = system.apply(c, optTrans)
                    if c is not None:
                        predTrees.append(c.tree)
                result = system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Optimization Finished.")

        print("Start predicting on test set")
        predTrees = []
        for sent in testSents:
            numTrans = system.numTransitions()

            c = system.initialConfiguration(sent)
            while not system.isTerminal(c):
                feat = getFeatures(c)
                pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = system.transitions[j]

                c = system.apply(c, optTrans)
            if c is not None:
                predTrees.append(c.tree)
        print("Store the test results.")
        Util.writeConll('result_nce_cube.conll', testSents, predTrees)

