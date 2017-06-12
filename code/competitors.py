from __future__ import print_function
import argparse
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.stats import threshold
from scipy.optimize import nnls
from sklearn.decomposition import LatentDirichletAllocation as LDA
import time

from compute_error import compute_error
from data_io import gen_data


class _ANLS_(object):
    def __init__(self , Y, A_init, A_true, epo = 200):
        self.A = A_init.copy()
        self.A_true = A_true.copy()
        self.Y = Y.copy()
        self.epo = epo
        self.time = 0

    def train(self):
        for i in range(self.epo):
            self.show_error()
            start = time.time()
            Z = self.decoding()
            self.A = self.Y * Z.transpose() * inv(Z * Z.transpose())
            end = time.time()
            self.time = self.time + end - start


    def decoding(self):
        D = self.A_true.shape[1]
        num_doc = self.Y.shape[1]
        Z = np.asmatrix(np.zeros((D, num_doc)))
        A = np.asarray(self.A.copy())
        Y = np.asarray(self.Y.copy())
        for i in range(num_doc):
            Yi = np.array(Y[:, i]).flatten()
            t, bla = nnls(A, Yi)
            Z[:, i] = np.asmatrix(t).transpose()
        Z = np.asmatrix(Z)
        return Z

    def show_error(self):
        error = compute_error(self.A, self.A_true)
        print('%f, %f' % (self.time, error))


class _MU_(object):
    def __init__(self , Y, A_init, A_true, epo = 1000):
        self.A = A_init.copy()
        self.A_true = A_true.copy()
        self.Y = Y.copy()
        self.Z = self.decoding()
        self.epo = epo
        self.time = 0

    def train(self):
        eps = 1e-10
        for i in range(self.epo):
            if i % 1 == 0:
                self.show_error()

            A = np.asarray(self.A.copy())
            Z = np.asarray(self.Z.copy())
            start = time.time()
            Z1 = np.multiply(Z, np.asarray(self.A.transpose() * self.Y))
            Z = np.divide(Z1, eps + np.asarray(self.A.transpose() * self.A * self.Z)) # + eps to avoid divided by 0
            self.Z = np.asmatrix(Z)
            A1 = np.multiply(A, np.asarray( self.Y * self.Z.transpose()))
            A = np.divide(A1, eps + np.asarray( self.A * self.Z * self.Z.transpose()))
            end = time.time()
            self.A = np.asmatrix(A)
            self.time = self.time + end - start

    def decoding(self):
        D = self.A_true.shape[1]
        num_doc = self.Y.shape[1]
        Z = np.asmatrix(np.zeros((D, num_doc)))
        for i in range(num_doc):
            Yi = np.array(self.Y[:, i].copy()).flatten()
            A = np.asarray(self.A.copy())
            t, bla = nnls(A, Yi)
            Z[:, i] = np.asmatrix(t).transpose()
        Z = np.asmatrix(Z)
        return Z

    def show_error(self):
        error = compute_error(self.A, self.A_true)
        print('%f, %f' % (self.time, error))


class _LDA_(object):
    def __init__(self, Y, A_init, A_true, sparsity = []):
        self.A = A_init.copy()
        self.A_true = A_true.copy()
        self.Y = Y.copy()
        self.Z = self.decoding()
        self.time = 0
        if sparsity:
            self.sparsity = sparsity
        else:
            self.sparsity = 5.0

    def train(self):
        D = self.A_true.shape[1]
        for i in range(20):
            self.show_error()

            start = time.time()
            prior = self.sparsity / np.float(self.A_true.shape[1])
            lda = LDA(n_topics=D, random_state=0, doc_topic_prior = prior, max_iter=i)
            lda.fit(self.Y.transpose())
            end = time.time()
            self.time = end - start
            self.A = np.asmatrix(lda.components_.transpose())

    def show_error(self):
        error = compute_error(self.A, self.A_true)
        print('%f, %f' % (self.time, error))

    def decoding(self):
        D = self.A_true.shape[1]
        num_doc = self.Y.shape[1]
        Z = np.asmatrix(np.zeros((D, num_doc)))
        for i in range(num_doc):
            Yi = np.array(self.Y[:, i].copy()).flatten()
            A = np.asarray(self.A.copy())
            t, bla = nnls(A, Yi)
            Z[:, i] = np.asmatrix(t).transpose()
        Z = np.asmatrix(Z)
        return Z


class _HALS_(object):
    def __init__(self , Y, A_init, A_true, epo = 10000):
        self.A = A_init.copy()
        self.A_true = A_true.copy()
        self.Y = Y.copy()
        self.Z = self.decoding()
        self.epo = epo
        self.time = 0

    def decoding(self):
        D = self.A_true.shape[1]
        num_doc = self.Y.shape[1]
        Z = np.asmatrix(np.zeros((D, num_doc)))

        for i in range(num_doc):
            Yi = np.array(self.Y[:, i].copy()).flatten()
            A = np.asarray(self.A.copy())
            t, bla = nnls(A, Yi)
            Z[:, i] = np.asmatrix(t).transpose()

        Z = np.asmatrix(Z)
        return Z


    def train(self):
        D = self.A_true.shape[1]
        for t in range(self.epo):
            self.show_error()
            for i in range(D):
                start = time.time()
                Yi = self.Y - self.A * self.Z + self.A[:, i] * self.Z[i, :]
                fi = self.A[:, i].copy()
                gi = self.Z[i, :].copy().transpose()
                self.A[:, i] = 1.0/(norm(gi) * norm(gi)) * threshold(Yi * gi, threshmin = 0)
                self.Z[i, :] = (1.0/(norm(fi) * norm(fi)) * threshold(Yi.transpose() * fi, threshmin = 0)).transpose()
                end = time.time()
                self.time = self.time + end - start

    def show_error(self):
        error = compute_error(self.A, self.A_true)
        print('%f, %f' % (self.time, error))


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
  '--seed',
  type=int,
  default=0,
  help='''Random seed (default 0)'''
)
parser.add_argument(
  '--method',
  type=int,
  default=1, # 1-4
  help='''Method to solve (1: ANLS, 2: MU, 3: LDA, 4: HALS; default 1)'''
)
parser.add_argument(
  '--A_type',
  type=str,
  default='load', # load, neg
  help='''Type of feature matrix A genereated (load, neg; default load)'''
)
parser.add_argument(
  '--topic_word_mat_file',
  type=str,
  default='../data/demo_L2_out.nips.100.A',
  help='''File containing the topic matrix (default: '../data/demo_L2_out.nips.100.A')'''
)
parser.add_argument(
  '--D',
  type=int,
  default=100,
  help='''Number of topics in the topic matrix (default: 100)'''
)
parser.add_argument(
  '--W',
  type=int,
  default=1000,
  help='''Number of words in the topic matrix (default: 1000)'''
)
parser.add_argument(
  '--num_sample',
  type=int,
  default=100 * 5 * 10,
  help='''Number of documents in the data matrix (default: 100 * 5 * 10)'''
)
parser.add_argument(
  '--x_type',
  type=str,
  default='ctm', # dir, ctm, uni
  help='''Type of weight matrix x genereated (dir, ctm, uni; default ctm)'''
)
parser.add_argument(
  '--noise',
  type=float,
  default=0.0,
  help='''Noise level (default 0.0; typical values 0.001)'''
)
parser.add_argument(
  '--dir_sparsity',
  type=float,
  default=5.0,
  help='''Sparsity in the DIR model (default: 5.0)'''
)
parser.add_argument(
  '--ctm_sigma',
  type=float,
  default=1.0,
  help='''Standard deviation in the Gaussian prior of the CTM model (default: 1.0)'''
)
parser.add_argument(
  '--uni_sparsity',
  type=float,
  default=5.0,
  help='''Sparsity in the uniform random prior (default: 5.0)'''
)
parser.add_argument(
  '--init_lratio',
  type=float,
  default=1.0,
  help='''Multiplier to the magnitude of the multiplicative error matrix in the initialization (default: 1.0)'''
)
parser.add_argument(
  '--init_nratio',
  type=float,
  default=0.0,
  help='''Multiplier to the magnitude of the additive error matrix in the initialization (default: 1.0)'''
)
FLAGS = parser.parse_args()
print("hyperparameters:")
print(FLAGS)

# run
np.random.seed(FLAGS.seed)

Ag, X, Y, N, A = gen_data(FLAGS)
if FLAGS.method == 1:
    print("method: ANLS")
    alg = _ANLS_(Y, A, Ag)
elif FLAGS.method == 2:
    print("method: MU")
    alg = _MU_(Y, A, Ag)
elif FLAGS.method == 3:
    print("method: LDA")
    alg = _LDA_(Y, A, Ag, FLAGS.dir_sparsity)
elif FLAGS.method == 4:
    print("method: HALS")
    alg = _HALS_(Y, A, Ag)

alg.train()