# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import time

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    row, column = X.shape
    result = np.zeros([row, row])
    for i in range(row):
        for j in range(row):
            xi = X[i, :]
            xj = X[j, :]
            dist = np.sqrt(np.dot(xi, xi) - 2 * np.dot(xi.T, xj) + np.dot(xj, xj))
            result[i, j] = dist
    return result


# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    row, column = X.shape
    # use X to create M
    x_squared = (X * X).sum(axis=1, keepdims=True)
    y_squared = x_squared.T
    result = x_squared - 2 * np.dot(X, X.T) + y_squared
    result[range(row), range(row)] = 0
    result = np.sqrt(result)
    return result


# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    row, column = X.shape
    result = np.zeros([column, column])
    for i in range(column):
        xi = X[:, i]
        mui = np.sum(xi) / row
        xni = xi - mui
        sigma_i = np.sqrt(np.dot(xni, xni) / (row - 1))
        for j in range(column):
            xj = X[:, j]
            muj = np.sum(xj) / row
            xnj = xj - muj
            sij = (np.dot(xni, xnj) / (row - 1))
            sigma_j = np.sqrt(np.dot(xnj, xnj) / (row - 1))
            if sigma_i == 0 or sigma_j == 0:
                result[i, j] = 0.0
            else:
                result[i, j] = sij / (sigma_i * sigma_j)
    return result


# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    rows, columns = X.shape
    mu_vector = (np.sum(X, axis=0)) / rows
    mu_matrix = mu_vector * np.ones([rows, 1])
    x = X - mu_matrix
    x_transposed = np.transpose(x)
    covariance = np.dot(x_transposed, x) / (rows - 1)
    x_squared = np.multiply(x, x)
    variance = np.sum(x_squared, axis=0) / (rows - 1)
    sigma = np.sqrt(variance)
    intermediate_result = np.outer(sigma, sigma)
    result = np.multiply(covariance, np.power(intermediate_result, -1))
    where_are_nans = np.isnan(result)
    result[where_are_nans] = 0
    return result


def main():
    print('starting comparing distance computation .....')
    np.random.seed(100)
    params = range(10, 141, 10)  # different param setting
    nparams = len(params)  # number of different parameters
    # 10 trials = 10 rows, each parameter is a column
    perf_dist_loop = np.zeros([10, nparams])
    perf_dist_cool = np.zeros([10, nparams])
    # 10 trials = 10 rows, each parameter is a column
    perf_corr_loop = np.zeros([10, nparams])
    perf_corr_cool = np.zeros([10, nparams])
    counter = 0
    for ncols in params:
        nrows = ncols * 10
        print("matrix dimensions: ", nrows, ncols)
        for i in range(10):
            X = np.random.rand(nrows, ncols)  # random matrix
            # compute distance matrices
            st = time.time()
            dist_loop = compute_distance_naive(X)
            et = time.time()
            perf_dist_loop[i, counter] = et - st  # time difference
            st = time.time()
            dist_cool = compute_distance_smart(X)
            et = time.time()
            perf_dist_cool[i, counter] = et - st
            # check if the two computed matrices are identical all the time
            # add assert after adding correct method
            assert np.allclose(dist_loop, dist_cool, atol=1e-10)
            np.savetxt('test-reports/dist-loop.txt', dist_loop, delimiter=',')
            np.savetxt('test-reports/dist-cool.txt', dist_cool, delimiter=',')

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i, counter] = et - st  # time difference
            #
            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i, counter] = et - st

            # check if the two computed matrices are identical all the time
            assert np.allclose(corr_loop, corr_cool, atol=1e-10)
            np.savetxt('test-reports/corr-loop.txt', corr_loop, delimiter=',')
            np.savetxt('test-reports/corr-cool.txt', corr_cool, delimiter=',')
        counter = counter + 1
    # mean time for each parameter setting (over 10 trials)
    mean_dist_loop = np.mean(perf_dist_loop, axis=0)
    mean_dist_cool = np.mean(perf_dist_cool, axis=0)
    std_dist_loop = np.std(perf_dist_loop, axis=0)  # standard deviation
    std_dist_cool = np.std(perf_dist_cool, axis=0)

    plt.figure(1)
    plt.errorbar(params,
                 mean_dist_loop[0:nparams],
                 yerr=std_dist_loop[0:nparams],
                 color='red',
                 label='Loop Solution for Distance Comp')
    plt.errorbar(params,
                 mean_dist_cool[0:nparams],
                 yerr=std_dist_cool[0:nparams],
                 color='blue',
                 label='Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.png')
    plt.savefig('test-reports/CompareDistanceCompFig.png')
    # plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareDistanceCompFig.png")
    #
    # # mean time for each parameter setting (over 10 trials)
    mean_corr_loop = np.mean(perf_corr_loop, axis=0)
    mean_corr_cool = np.mean(perf_corr_cool, axis=0)
    std_corr_loop = np.std(perf_corr_loop, axis=0)  # standard deviation
    std_corr_cool = np.std(perf_corr_cool, axis=0)

    plt.figure(2)
    plt.errorbar(params,
                 mean_corr_loop[0:nparams],
                 yerr=std_corr_loop[0:nparams],
                 color='red',
                 label='Loop Solution for Correlation Comp')
    plt.errorbar(params,
                 mean_corr_cool[0:nparams],
                 yerr=std_corr_cool[0:nparams],
                 color='blue',
                 label='Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.png')
    plt.savefig('test-reports/CompareCorrelationCompFig.png')
    # plt.show()  # uncomment this if you want to see it right way
    print("result is written to CompareCorrelationCompFig.png")


if __name__ == "__main__":
    main()
