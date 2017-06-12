import numpy as np
from numpy.linalg import norm

def compute_error(A_in, Ag_in):
    A = A_in
    Ag = Ag_in

    #reallign
    D = A.shape[1]
    inner = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            inner[i, j] = np.asscalar(A[:, i].transpose() * Ag[:, j] )/(norm(A[:, i]) * norm(Ag[:, j]))

    max = np.argmax(inner, axis = 0)
    P = np.asmatrix(np.zeros((D, D)))
    for i in range(D):
        P[i, max[i]] = 1

    # print "normalize the rows of A and A^*"
    inv_norm_A = np.asarray(1.0 / np.apply_along_axis(norm, 0, A))
    A = A * np.diag(inv_norm_A)
    inv_norm_Ag = np.asarray(1.0 / np.apply_along_axis(norm, 0, Ag))
    Ag = Ag * np.diag(inv_norm_Ag)

    u = np.asmatrix(np.ones((1, D)))
    #for each A_i^* we try to find the A_i that is closest to A_i^*
    error = 0
    for i in range(D):
        Ag_i = Ag[:, i]
        inner_product = np.asmatrix(Ag_i.transpose() * A)
        norm_A = np.asmatrix(np.diag(A.transpose() * A))
        z = np.divide(inner_product, norm_A).transpose()
        z = np.asarray(z).flatten().transpose()
        scalar = np.diag(z)
        As = A * scalar
        diff = np.apply_along_axis(norm, 0, As - Ag_i * u)
        # min_idx = np.argmin(diff)
        # print 'for Ag_%d: A_%d' % (i, min_idx)
        difmin = np.amin(diff)
        difmin = difmin * difmin
        error = error + difmin

    return error
