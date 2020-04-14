import numpy as np
_EPS = 1e-10


def kernalign(matrix1, matrix2_or_y, indices=None):
    """Computes the kernel-kernel or kernel-target alignment [1]_:
    .. math::
        <matrix1, matrix2> / \\sqrt{ <matrix1, matrix1> <matrix2, matrix2> }
    Parameters
    ----------
    matrix1 : np.ndarray of shape (n, n)
        The Gram matrix.
    matrix2_or_target : np.ndarray of shape (n, n) or (n,)
        Either a second Gram matrix or a target vector. If the shape is (n,),
        then matrix2 is the outer product :math:`target target^\\top`.
    indices : ordered collection
        Indices that define the empirical sample.
    Returns
    -------
    alignment : float
        The alignment.
    References
    ----------
    .. [1] N. Cristianini et al. *On Kernel-Target Alignment*, 2001.
    """
    if matrix2_or_y.ndim == 1:
        matrix2 = np.outer(matrix2_or_y, matrix2_or_y).ravel()
    else:
        matrix2 = matrix2_or_y.ravel()

    if indices is not None:
        matrix1 = matrix1[np.ix_(indices, indices)]
        matrix2 = matrix2[np.ix_(indices, indices)]

    matrix1, matrix2 = matrix1.ravel(), matrix2.ravel()
    dot11 = np.dot(matrix1, matrix1)
    dot12 = np.dot(matrix1, matrix2)
    dot22 = np.dot(matrix2, matrix2)
    return dot12 / (np.sqrt(dot11 * dot22) + _EPS)

def get_feature(file):
    m = np.shape(file)[0]
    n = np.shape(file)[1]
    data = np.zeros((m, n-1))
    for index in range(m):
        data[index] = file[index][1:]
    return data


np.set_printoptions(suppress = True)

dataset_name = ['Antifp_Main', 'Antifp_DS1', 'Antifp_DS2']
for ds in range(3):
    name_ds = dataset_name[ds]
    print('dataset:', name_ds)
    methods_name = ['188-bit', 'AAC', 'ASDC', 'CKSAAP', 'CTD', 'DPC']
    for it in range(6):
        name = methods_name[it]
        print(name + ':')

        f1 = np.loadtxt('D:/Study/Bioinformatics/AFP/kernel_matrix/' + name_ds + '/KM_train_tanimoto/KM_tanimoto_' + name + '_train.csv', delimiter = ',')
        f2 = np.loadtxt('D:/Study/Bioinformatics/AFP/feature_matrix/' + name_ds + '/train_label.csv', delimiter = ',')

        gram = f1
        Y = f2
        weight = kernalign(gram, Y)
        print(weight)