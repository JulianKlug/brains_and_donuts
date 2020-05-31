import numpy as np
from scipy.linalg import norm, eigh

def dot(x, dir):
    return 2 ** 31 if len(x) == 1 else x[0] * dir[0] + x[1] * dir[1]


def orth_proj(x, dir):
    return dot(x, dir) * dir


def wasserstein(barcode1, barcode2, M, ord=1):
    """
    Approximate Sliced Wasserstein distance between two barcodes
    :param barcode1:
    :param barcode2:
    :param M: the approximation factor, bigger M means more accurate result
    :param ord: p-Wassertein distance to use
    :return:
    """
    diag = np.array([np.sqrt(2), np.sqrt(2)])
    b1 = list(barcode1)
    b2 = list(barcode2)
    for bar in barcode1:
        b2.append(orth_proj(bar, diag))
    for bar in barcode2:
        b1.append(orth_proj(bar, diag))
    b1 = np.array(b1, copy=False)
    b2 = np.array(b2, copy=False)
    s = np.pi / M
    theta = -np.pi / 2
    sw = 0
    for i in range(M):
        dir = np.array([np.cos(theta), np.sin(theta)])
        v1 = np.sort(np.dot(b1, dir))
        v2 = np.sort(np.dot(b2, dir))
        sw += s * norm(v1 - v2, ord)
        theta += s
    return sw / np.pi

