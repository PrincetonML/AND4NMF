import sys
import numpy as np
from numpy.linalg import pinv
from scipy.stats import threshold
import time
import argparse

from compute_error import compute_error
from data_io import gen_data

class _AND_(object):
    def __init__(self , Y, A_init, A_true, init_alpha, inner_epo = 50, outer_epo = 200, decrease_rate = 1.1, FLAGS = []):
        """If no groundtruth known, set A_true to be [].
        If used in sklearn format, then set Y = [], A_true = [], and then use model.fit(Y), and get the result from model.components_"""
        self.A = A_init.copy()
        self.A_true = np.copy(A_true)
        self.Y = Y.copy()
        self.alpha = init_alpha
        self.inner_epo = inner_epo
        self.outer_epo = outer_epo
        self.decrease_rate = decrease_rate
        self.time = 0
        self.FLAGS = FLAGS

    def fit(self, Y):
        "sklearn like method"
        self.Y = np.transpose(Y) # need to transpose to match the convention of sklearn
        self.train()
        self.components_ = self.A.transpose()

    def train(self):
        for i in range(self.outer_epo):
            self.alpha = self.alpha / self.decrease_rate
            self.A = self._update()

    def _update(self):
        D = self.A.shape[1]
        eta = 0.5 /np.float(D) # to tune

        A_t = self.A.copy()
        if self.A_true.any():
            self.show_error()
            sys.stdout.flush()

        start = time.time()
        A_inv = pinv(self.A)
        Z = threshold(A_inv * self.Y, threshmin = self.alpha)
        for i in range(self.inner_epo):
            A_t = A_t + eta * (self.Y * Z.transpose() - A_t * Z * Z.transpose())
        end = time.time()
        self.time = self.time + end - start
        return A_t

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
  '--thres_decay',
  type=float,
  default=1.1,
  help='''Decay factor of the threshold (default 1.1)'''
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
parser.add_argument(
  '--init_thres',
  type=float,
  default=0.1,
  help='''inital value for the threshold (default 0.1)'''
)
parser.add_argument(
  '--outer_epo',
  type=int,
  default=200,
  help='''Number of outer loops (default 200)'''
)
parser.add_argument(
  '--inner_epo',
  type=int,
  default=50,
  help='''Number of inner loops (default 50; typical values: 50 for no noise, 100 for noise)'''
)
FLAGS = parser.parse_args()
print("hyperparameters:")
print(FLAGS)

# run
np.random.seed(FLAGS.seed)

Ag, X, Y, N, A = gen_data(FLAGS)
print("method: AND")
AND = _AND_(Y, A, Ag, FLAGS.init_thres, FLAGS.inner_epo, FLAGS.outer_epo, FLAGS.thres_decay, FLAGS)
AND.train()
