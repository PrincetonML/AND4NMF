from __future__ import print_function
import numpy as np

def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
        Rows are scores for each class.
        Columns are predictions (samples).
        """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

def gen_cor_topic_vector(K, sigma=1.0):
    """Generate topic vectors"""
    mu = np.zeros(K)
    diag_Sigma = np.random.rand(K, K)
    off_Sigma = (np.random.rand(K, K) - 0.5) * sigma / K
    off_Sigma = off_Sigma - np.diag(np.diag(off_Sigma))
    Sigma = diag_Sigma + off_Sigma
    logit = mu + np.dot(Sigma, np.random.randn(K))
    topics = softmax(logit)
    print('max sig, off, topic: %.3f %.3f %.3f' % (np.max(np.diag(Sigma)), np.max(np.abs(off_Sigma)), max(topics)) )

    return topics

def gen_data(FLAGS):
    """
    :param FLAGS: contains the following fields
    D: number of topics
    num_sample: number of samples
    init_lratio, init_nratio: change the ground-truth topic matrix Ag
    noise: noise level to the data matrix Y
    A_type: 'load' or 'neg'
      if 'load': need topic_word_mat_file
      if 'neg': need W, number of words
    x_type: 'dir' 'ctm' 'uni'
      if 'dir': need dir_sparsity
      if 'ctm': need ctm_sigma
      if 'uni': need uni_sparsity
    :return: ground-truth topic matrix Ag, coefficient matrix X, data Y, noise N, initialization A
    """
    D = FLAGS.D
    l = 1/np.sqrt(D)
    num_sample = FLAGS.num_sample

    init_lratio = FLAGS.init_lratio
    init_nratio = FLAGS.init_nratio

    if FLAGS.A_type == 'load':
        topic_word_mat_file = FLAGS.topic_word_mat_file
        Ag = np.asmatrix(np.loadtxt(topic_word_mat_file))
        SEZ = np.asmatrix(np.random.rand(D, D) - 0.5) * l * init_lratio
        AN = np.asmatrix(np.random.rand(Ag.shape[0], D) - 0.5) * l * init_nratio
        A = Ag * (np.asmatrix(np.identity(D)) + SEZ) + AN
    elif FLAGS.A_type == 'neg':
        W = FLAGS.W
        Ag = np.asmatrix(np.random.rand(W, D)  - 0.5 )
        SEZ = np.asmatrix(np.random.rand(D, D) - 0.5 ) * l * init_lratio
        AN = np.asmatrix(np.random.rand(Ag.shape[0], D) - 0.5) * l * init_nratio
        A = Ag * (np.asmatrix(np.identity(D)) + SEZ) + AN
    else:
        print("unknown A_type ", FLAGS.A_type)
    print('A^* shape %d %d' % (Ag.shape[0], Ag.shape[1]))

    if FLAGS.x_type == 'dir':
        dir_a = FLAGS.dir_sparsity /np.float32(D)
        X = np.asmatrix(np.random.dirichlet(([ dir_a for j in range(D)]), num_sample )).transpose()
    elif FLAGS.x_type == 'ctm':
        X = np.zeros((D, num_sample))
        for i in range(num_sample):
            X[:, i] = gen_cor_topic_vector(D, FLAGS.ctm_sigma)
        X = np.asmatrix(X)
    elif FLAGS.x_type == 'uni':
        X = np.random.rand(D, num_sample)
        X = np.greater(X, 1 - 1.0 / FLAGS.uni_sparsity )
        X = np.asmatrix(X.astype(dtype='float'))
    else:
        print('unknown type of x: ', FLAGS.x_type)
    print('X shape %d %d' % (X.shape[0], X.shape[1]))

    if FLAGS.noise > 0:
        N = np.asmatrix(np.random.normal(0, FLAGS.noise / Ag.shape[0], (Ag.shape[0], X.shape[1])))
    else:
        N = np.zeros((Ag.shape[0], X.shape[1]))
    Y = Ag * X + N

    # output data stats
    r = np.max( np.sum(X, 0) )
    X_m2 = np.dot(X, np.transpose(X)) / num_sample
    k = np.max(np.diag(X_m2)) * D / 2.0
    off_X_m2 = X_m2 - np.diag(np.diag(X_m2))
    m = np.max(off_X_m2) * D * D
    x_lambda = np.min(np.diag(X_m2)) * D / k
    m_bound = k * D * np.power(x_lambda,4) / np.power(r,5)
    print('r=%f, k=%f, m=%f, lambda=%f; bound=%f' % (r, k, m, x_lambda, m_bound))
    print('init multiplicative error ||E||_2 = %f' % np.linalg.norm(SEZ, ord=2))
    print('init additative error (column norm) ||N||_1 = %f' % np.linalg.norm(AN, ord=1))

    return Ag, X, Y, N, A