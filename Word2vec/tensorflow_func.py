import tensorflow as tf
import numpy as np

def nce_loss(inputs,weights,biases,labels,sample,unigram_prob):

    (rowShape, colShape) = inputs.get_shape().as_list()

    sample = np.reshape(sample, (len(sample), 1))
    Biasx = tf.nn.embedding_lookup(biases, sample)
    Biasx = tf.transpose(Biasx)


    Biaso = tf.nn.embedding_lookup(biases, labels)
    Biaso = tf.transpose(Biaso)


    Pr_Wo = tf.nn.embedding_lookup(np.asarray(unigram_prob), labels)


    kPr_Wo = tf.multiply(tf.to_float(Pr_Wo), 1.0 * rowShape)


    log_kPr_Wo = tf.transpose(tf.log(kPr_Wo))


    words = tf.nn.embedding_lookup(weights, labels)
    words = tf.reshape(words, [-1, colShape])
    words = tf.matmul(inputs, tf.transpose(words))
    words = tf.add(words, Biaso)

    subA = tf.subtract(words, log_kPr_Wo)

    sig_subA = tf.sigmoid(subA)

    A = tf.diag_part(tf.log(sig_subA))
    A = tf.reshape(A, [rowShape, 1])

    Wx = tf.nn.embedding_lookup(weights, sample)
    Wx = tf.reshape(Wx, [-1, colShape])

    (rowSample, colSample) = Wx.get_shape().as_list()

    Pr_Wx = tf.nn.embedding_lookup(np.asarray(unigram_prob), sample)

    kPr_Wx = tf.multiply(tf.to_float(Pr_Wx), 1.0 * rowSample)


    log_kPr_Wx = tf.transpose(tf.log(kPr_Wx))


    b = tf.matmul(inputs, tf.transpose(Wx))
    b = tf.add(b, Bx)


    subB = tf.subtract(b, log_kPr_Wx)

    sigmoid_subB = tf.sigmoid(subB)

    t_array = tf.subtract(1.0, sigmoid_subB)

    B = tf.reduce_sum(tf.log(t_array), -1, keep_dims=True)

    loss = tf.multiply(tf.add(A, B), -1.0)
    print ('loss ---> ', loss)

    return loss



def cross_entropy_loss(inputs, true_w):

    (rowShape, colShape) = inputs.get_shape().as_list()

    # Form A, B

    transpose = tf.transpose(true_w)

    product = tf.matmul(inputs, transpose)

    # Now i have to take all diagnol elements

    diagPart = tf.diag_part(product)
    exp = tf.exp(diagPart, name=None)
    log_m = tf.log(exp, name=None)
    A = tf.reshape(log_m, [rowShape, 1])
    bExp = tf.exp(product, name=None)
    sum_m = tf.reduce_sum(bExp, -1, keep_dims=True)
    B = tf.log(sum_m, name=None)

    # Here we have done B-A because we are

    return tf.subtract(B, A)





			