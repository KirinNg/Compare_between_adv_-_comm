import numpy as np

A = np.load("A.npy")
A = A[0, :, :, :]

for i in range(149):
    for j in range(149):
        a, b, c, d = A[i * 2][j * 2], A[i * 2 + 1][j * 2], A[i * 2][j * 2 + 1], A[i * 2 + 1][j * 2 + 1]
        A[i * 2][j * 2], A[i * 2 + 1][j * 2], A[i * 2][j * 2 + 1], A[i * 2 + 1][j * 2 + 1] = d, b, a, c

A = A[np.newaxis, :, :, :]
np.save("r_A.npy", A)
