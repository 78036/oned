import timeit

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits

import ComputeMatrices


def do_iris():
    iris = load_breast_cancer()
    naive_distance = ComputeMatrices.compute_distance_naive(iris.data)
    np.savetxt('test-reports/iris_naive_distance.txt', naive_distance, delimiter=',')
    smart_distance = ComputeMatrices.compute_distance_smart(iris.data)
    np.savetxt('test-reports/iris_smart_distance.txt', smart_distance, delimiter=',')
    assert np.allclose(naive_distance, smart_distance, atol=1e-10)
    naive_correlation = ComputeMatrices.compute_correlation_naive(iris.data)
    np.savetxt('test-reports/iris_naive_correlation.txt', naive_correlation, delimiter=',')
    smart_correlation = ComputeMatrices.compute_correlation_smart(iris.data)
    np.savetxt('test-reports/iris_smart_correlation.txt', smart_correlation, delimiter=',')
    assert np.allclose(naive_correlation, smart_correlation, atol=1e-10)


def do_breast_cancer():
    cancer = load_breast_cancer()
    naive_distance = ComputeMatrices.compute_distance_naive(cancer.data)
    np.savetxt('test-reports/cancer_naive_distance.txt', naive_distance, delimiter=',')
    smart_distance = ComputeMatrices.compute_distance_smart(cancer.data)
    np.savetxt('test-reports/cancer_smart_distance.txt', smart_distance, delimiter=',')
    assert np.allclose(naive_distance, smart_distance, atol=1e-10)
    naive_correlation = ComputeMatrices.compute_correlation_naive(cancer.data)
    np.savetxt('test-reports/cancer_naive_correlation.txt', naive_correlation, delimiter=',')
    smart_correlation = ComputeMatrices.compute_correlation_smart(cancer.data)
    np.savetxt('test-reports/cancer_smart_correlation.txt', smart_correlation, delimiter=',')
    assert np.allclose(naive_correlation, smart_correlation, atol=1e-10)


def do_digits():
    digits = load_digits()
    naive_distance = ComputeMatrices.compute_distance_naive(digits.data)
    np.savetxt('test-reports/digits_naive_distance.txt', naive_distance, delimiter=',')
    smart_distance = ComputeMatrices.compute_distance_smart(digits.data)
    np.savetxt('test-reports/digits_smart_distance.txt', smart_distance, delimiter=',')
    assert np.allclose(naive_distance, smart_distance, atol=1e-10)
    naive_correlation = ComputeMatrices.compute_correlation_naive(digits.data)
    np.savetxt('test-reports/digits_naive_correlation.txt', naive_correlation, delimiter=',')
    smart_correlation = ComputeMatrices.compute_correlation_smart(digits.data)
    np.savetxt('test-reports/digits_smart_correlation.txt', smart_correlation, delimiter=',')
    # assert np.allclose(naive_correlation, smart_correlation, atol=1e-10)


def main():
    print("Hello, world!")
    print("do iris")
    print(timeit.Timer(do_iris()).timeit(number=10))
    print("do breast cancer")
    print(timeit.Timer(do_breast_cancer()).timeit(number=10))
    print("do digits")
    print(timeit.Timer(do_digits()).timeit(number=10))
    print("bye")


if __name__ == "__main__":
    main()
