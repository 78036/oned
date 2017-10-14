# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import time

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# this is just me trying to learn
def compute_stupid_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    M = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            xi = X[i, :]
            xj = X[j, :]
            # dist = np.dot(xi.T, xj)
            dist = np.dot(xi.T, xi)
            M[i, j] = dist
    return M


# this is just me trying to learn
def compute_stupid_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    # use X to create M
    # a = np.dot(X, X.T)
    a = np.sum(X ** 2, axis=1)
    b = np.tile(a, (D, D))
    return b


# get xi
def get_sum_x_i(input_matrix):
    number_of_rows = input_matrix.shape[0]
    number_of_columns = input_matrix[0].shape[0]
    result = np.zeros(number_of_columns)
    for i in range(number_of_columns):
        for j in range(number_of_rows):
            result[i] += input_matrix[j, i]
    return result


# get sii
def get_s_i_i(input_matrix, mu, row_number):
    n_max = input_matrix.shape[0]
    intermediate_subtract = 0.0
    for n in range(n_max):
        intermediate_subtract += (input_matrix[n][row_number] - mu[row_number]) ** 2
    result = intermediate_subtract / (n_max - 1)
    assert np.abs(result) < 1
    return result


# get sij
def get_s_i_j(input_matrix, mu, row_number, column_number):
    n_max = input_matrix.shape[0]
    intermediate_subtract = 0.0
    for n in range(n_max):
        intermediate_subtract += (
            (input_matrix[n][row_number] - mu[row_number]) * (input_matrix[n][row_number] - mu[column_number]))
    result = intermediate_subtract / (n_max - 1)
    assert np.abs(result) < 1
    return result


# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    M = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            xi = X[i, :]
            xj = X[j, :]
            # dist = 0.0  # a placetaker line,
            # you have to change it to distance between xi and xj
            # sum = 0.0
            # for increment in range(D):
            #     sum = sum + np.square(X[i, increment].T - X[j, increment].T)
            # dist = np.sqrt(sum)
            # dist = np.linalg.norm(xi - xj)
            dist = np.sqrt(np.dot(xi, xi) - 2 * np.dot(xi.T, xj) + np.dot(xj, xj))
            M[i, j] = dist
    return M


# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    # use X to create M
    x_squared = (X * X).sum(axis=1)[:, np.newaxis]
    y_squared = x_squared.T
    result = x_squared - 2 * np.dot(X, X.T) + y_squared
    np.maximum(result, 0, out=result)
    result = np.sqrt(result)
    return result


# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    # use X to create M
    M = np.zeros([D, D])
    sum_x_i = get_sum_x_i(X)
    sample_mean = sum_x_i / N
    assert sample_mean.shape[0] == D
    M = np.zeros([D, D])
    sij = np.zeros([D, D])
    for i in range(D):
        s_i_i = get_s_i_i(input_matrix=X, mu=sample_mean, row_number=i)
        for j in range(D):
            s_i_j = get_s_i_j(input_matrix=X, mu=sample_mean, row_number=i, column_number=j)
            if (i == j):
                assert s_i_i == s_i_j
            sij[i][j] = s_i_j
            xi = X[:, i]
            xj = X[:, j]
            if sij[i][i] != 0 or sij[j][j] != 0:
                corr = 0
            else:
                corr = sij[i][j] / (np.sqrt(sij[i][i]) * np.sqrt(sij[j][j]))
            # a placetaker line,
            # you have to change it to correlation between xi and xj
            M[i, j] = corr
    return M


# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    # N = X.shape[0]  # num of rows
    # D = X[0].shape[0]  # num of cols
    #
    # # use X to create M
    # M = np.corrcoef(X)
    # return M
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    S = np.zeros([D, D])
    for i in range(D):
        for j in range(D):
            m = sum([X[n, i] for n in range(N)]) / N
            S[i, j] = sum([(X[n, i] - m) * (X[n, j] - m) for n in range(N)]) / (N - 1.)
    R = np.zeros([D, D])
    for i in range(D):
        for j in range(D):
            ss = np.sqrt(S[i, i]) * np.sqrt(S[j, j])
            R[i, j] = S[i][j] / ss if ss != 0 else 0
    return R


def main():
    print('starting comparing distance computation .....')
    np.random.seed(100)
    params = range(10, 141, 10)  # different param setting
    nparams = len(params)  # number of different parameters

    # 10 trials = 10 rows, each parameter is a column
    perf_stupid_loop = np.zeros([10, nparams])
    perf_stupid_cool = np.zeros([10, nparams])

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
            stupid_loop = compute_stupid_naive(X)
            et = time.time()
            perf_stupid_loop[i, counter] = et - st  # time difference

            st = time.time()
            stupid_cool = compute_stupid_smart(X)
            et = time.time()
            perf_stupid_cool[i, counter] = et - st

            # check if the two computed matrices are identical all the time
            # add assert after adding correct method
            # assert np.allclose(stupid_loop, stupid_cool, atol=1e-06)
            np.savetxt('test-reports/stupid-loop.txt', stupid_loop, delimiter=',')
            np.savetxt('test-reports/stupid-cool.txt', stupid_cool, delimiter=',')

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
            assert np.allclose(dist_loop, dist_cool, atol=1e-06)
            np.savetxt('test-reports/dist-loop.txt', dist_loop, delimiter=',')
            # np.savetxt('test-reports/dist-cool.txt', dist_cool, delimiter=',')

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i, counter] = et - st  # time difference

            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i, counter] = et - st

            # check if the two computed matrices are identical all the time
            # assert np.allclose(corr_loop, corr_cool, atol=1e-06)
            np.savetxt('test-reports/corr-loop.txt', corr_loop, delimiter=',')
            np.savetxt('test-reports/corr-cool.txt', corr_cool, delimiter=',')

        counter = counter + 1

    # mean time for each parameter setting (over 10 trials)
    mean_stupid_loop = np.mean(perf_dist_loop, axis=0)
    mean_stupid_cool = np.mean(perf_dist_cool, axis=0)
    std_stupid_loop = np.std(perf_dist_loop, axis=0)  # standard deviation
    std_stupid_cool = np.std(perf_dist_cool, axis=0)

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

    # mean time for each parameter setting (over 10 trials)
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

    plt.figure(3)
    plt.errorbar(params,
                 mean_stupid_loop[0:nparams],
                 yerr=std_stupid_loop[0:nparams],
                 color='red',
                 label='Loop Solution for Stupid Comp')
    plt.errorbar(params,
                 mean_stupid_cool[0:nparams],
                 yerr=std_stupid_cool[0:nparams],
                 color='blue',
                 label='Matrix Solution for Stupid Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Stupid Computation Methods')
    plt.legend()
    plt.savefig('CompareStupidCompFig.png')
    plt.savefig('test-reports/CompareStupidCompFig.png')
    # plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareStupidCompFig.png")


if __name__ == "__main__":
    main()
