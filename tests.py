import numpy as np
import pytest

from main import matmul, matmul4r


def check_equal(np_result, matmul_result, matmul4r_result):
    assert np.array_equal(np_result, matmul_result)
    assert np.array_equal(np_result, matmul4r_result)


def compute(mat_a, mat_b):
    return np.matmul(mat_a, mat_b) % 2, matmul(mat_a, mat_b, binary=True), matmul4r(mat_a, mat_b)


def test_trivial_case():
    mat_a = np.array([[0]])
    mat_b = np.array([[0]])

    np_result, matmul_result, matmul4r_result = compute(mat_a, mat_b)
    check_equal(np_result, matmul_result, matmul4r_result)


def test_random_case():
    n = np.random.randint(1, 101)
    mat_a = np.random.randint(0, 2, (n, n))
    mat_b = np.random.randint(0, 2, (n, n))

    np_result, matmul_result, matmul4r_result = compute(mat_a, mat_b)
    check_equal(np_result, matmul_result, matmul4r_result)


def test_non_square():
    n = np.random.randint(1, 101)
    mat_a = np.random.randint(0, 2, (n, n+1))
    mat_b = np.random.randint(0, 2, (n+1, n))

    with pytest.raises(Exception):
        _ = matmul4r(mat_a, mat_b)


def test_non_binary():
    n = np.random.randint(1, 101)
    mat_a = np.random.randint(0, 2, (n, n))
    mat_b = np.random.randint(0, 2, (n, n))
    mat_a[0, 0] = 2

    # with pytest.raises(Exception):
    _ = matmul4r(mat_a, mat_b)
