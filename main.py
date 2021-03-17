import numpy as np
from functools import reduce


def check_matrices(mat_a, mat_b, square=False, binary=False):
    """
    This method checks the size and the content of two passed matrices
    """
    assert len(mat_a.shape) == 2
    assert len(mat_b.shape) == 2
    rows_a, cols_a = mat_a.shape
    rows_b, cols_b = mat_b.shape
    assert cols_a == rows_b
    if square:
        assert rows_a == cols_a and rows_b == cols_b
    if binary:
        assert np.array_equal(mat_a, mat_a.astype(bool))
        assert np.array_equal(mat_b, mat_b.astype(bool))


def matmul(mat_a, mat_b, binary=False):
    """
    This method performs the matrix multiplication in a standard way with complexity O(n^3).
    """
    check_matrices(mat_a, mat_b, binary=binary)
    rows_a, cols_a = mat_a.shape
    rows_b, cols_b = mat_b.shape

    result = np.zeros((rows_a, cols_b), dtype='int32')
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_b):
                result[i][j] += mat_a[i][k] * mat_b[k][j]
                if binary:
                    result[i][j] = result[i][j] % 2
    return result


def expand_with_zeros(element, k):
    """
    This method takes a sequence as an input and checks if it fits the given length k.
    If the length of sequence is smaller, zeros are concatenated to the end of the sequence.
    """
    return element if len(element) == k else np.array(list(element) + [0] * (k - len(element)))


def split_matrix(mat, vertical=False):
    """
    This method splits the matrix in a vertical or horizontal way as it is done in the Method of Four Russians.
    (check README.md)
    """
    n = len(mat)
    k = int(np.log2(n))

    if vertical:
        mat_new = [
            [expand_with_zeros(mat[i:i + k, j], k) for j in range(n)]
            for i in range(0, n, k)
        ]
    else:
        mat_new = [
            [expand_with_zeros(mat[i][j:j + k], k) for j in range(0, n, k)]
            for i in range(n)
        ]
    return np.array(mat_new)


def xor(a, b):
    """
    This method calculates xor between two binary numbers.
    """
    return int(bool(a) ^ bool(b))


def xor_array(arr):
    """
    This method calculates xor of an array of binary numbers.
    """
    return reduce(lambda a, b: xor(a, b), arr, 0)


def scalar_mult_mod_2(arr_a, arr_b):
    """
    This method calculates the scalar product of two vectors mod 2.
    """
    return xor_array([a * b for a, b in zip(arr_a, arr_b)])


def make_count_matrix(mat_a_split, mat_b_split):
    """
    This method precomputes the matrix of all possible binary tuples from two input sequences,
    as it is done in the Method of Four Russians (check README.md).
    """
    mat_a_unique = list(set([tuple(el) for row in mat_a_split for el in row]))
    mat_b_unique = list(set([tuple(el) for row in mat_b_split for el in row]))

    count_matrix = {}
    for un_a in mat_a_unique:
        for un_b in mat_b_unique:
            count_matrix[(un_a, un_b)] = scalar_mult_mod_2(un_a, un_b)
    return count_matrix


def row_x_col(row, col, count_matrix):
    """
    This method calculates multiplication row * col, taking element-wise products from count matrix.
    """
    return xor_array([count_matrix[(tuple(row_elem), tuple(col_elem))]
                      for row_elem, col_elem in zip(row, col)])


def matmul4r(mat_a, mat_b):
    """
    This method performs the matrix multiplication as it is done in the Method of Four Russians (check README.md).
    """
    check_matrices(mat_a, mat_b, square=True, binary=True)
    n = mat_a.shape[0]
    if n == 1:
        return np.array([[mat_a[0][0] * mat_b[0][0]]])

    mat_a_new = split_matrix(mat_a)
    mat_b_new = split_matrix(mat_b, vertical=True)

    count_matrix = make_count_matrix(mat_a_new, mat_b_new)

    result = np.zeros((n, n), dtype='int32')
    for i in range(n):
        for j in range(n):
            result[i][j] = row_x_col(mat_a_new[i], mat_b_new[:, j], count_matrix)
    return result
