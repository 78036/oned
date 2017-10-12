This project uses Python 3.

Make sure you have python 3 installed.
I use Python 3.6.
You can check by running `python3 --version` in your shell.
I recommend that you use python 3 virtual environment.
Once you install python3 venv, you can create a virtual environment.
For example, on Ubuntu or Fedora, you can run `python3 -m venv venv`.
Then, go to the venv using something like `source venv/bin/activate`
Make sure your pip is up to date using `pip install --upgrade pip`
Then, you can install dependencies from the requirements.txt file using `pip install -r requirements.txt`.
To run, simply call python ComputeMatrices.py

I calculated the naive distance and naive correlation.
However, I still don't fully grasp how to do vectorization.

What was I asked to do?

I was asked to calculate distance matrix and correlation matrix.
It is important because doing this exercise will teach us about vectorization and broadcasting.

I implemented the loop version in very simple code.

I assume the loop versions are correct.
Therefore, I cannot test for its accuracy.

I learned that I don't know enough numpy.


```

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
```


Naive correlation
```
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
            if(i == j):
                assert s_i_i == s_i_j
            sij[i][j] = s_i_j
            xi = X[:, i]
            xj = X[:, j]
            corr = sij[i][j] / (np.sqrt(sij[i][i]) * np.sqrt(sij[j][j]))
            # a placetaker line,
            # you have to change it to correlation between xi and xj
            M[i, j] = corr
    return M
```


[![CircleCI](https://circleci.com/gh/7165015874/78036onetest.svg?style=svg)](https://circleci.com/gh/7165015874/78036onetest)

The example circle ci is at https://circleci.com/gh/7165015874/78036onetest

This is not a change.
