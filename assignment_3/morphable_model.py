"""
Section 2 of Assignment 3
Morphable Model

Point Cloud generation PCA equation:
G(\alpha, \delta) = \mu_{id} + E_{id}[\alpha * \sigma_{id}]
    + \mu_{exp} + E_{exp}[\delta * \sigma_{exp}]

, where
\mu_{id} is of shape (N, 3) - mean facial identity
\mu_{exp} is of shape (N, 3) - mean facial expression
E_{id} is of shape (N, 3, 30) - principal components for facial identity
E_{exp} is of shape (N, 3, 20) - principal components for facial expression
"""
import h5py
import numpy as np

from assignment_3.utils.data_def import Mesh


def morphable_model(file):
    """
    Constructs the PCA model using the equation above.
    :param file: path to file
    :return: a Mesh
    """
    # read the file using h5py
    bfm = h5py.File(file, 'r')

    # get the facial identity
    mean_id = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
    pca_basis_id = np.asarray(bfm['shape/model/pcaBasis'],
                              dtype=np.float32).reshape((-1, 3, 199))[:, :, :30]
    pca_var_id = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)[:30]

    # get the facial expression
    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3))
    pca_basis_exp = np.asarray(bfm['expression/model/pcaBasis'],
                               dtype=np.float32).reshape((-1, 3, 100))[:, :, :20]
    pca_var_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)[:20]

    # sample alpha and delta
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)

    # compute principal components
    pc_id = np.dot(pca_basis_id, alpha * np.sqrt(pca_var_id))
    pc_exp = np.dot(pca_basis_exp, delta * np.sqrt(pca_var_exp))

    # compute face geometry
    face = mean_id + pc_id + mean_exp + pc_exp

    # get the mean face color
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
    # get the triangles
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

    # return the Mesh
    return Mesh(face, mean_tex, triangles)
