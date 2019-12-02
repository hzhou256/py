import numpy as np


m = int(input())
n = int(input())
A = np.zeros((m, n))
Σ = np.zeros((m, n))
for i in range(m):
    A[i] = list(map(int, input().strip().split()))
B = A.T
λ, v = np.linalg.eig(np.dot(B, A))
λ1, u = np.linalg.eig(np.dot(B, A))
while i < m and i < n:
    Σ[i][i] = np.sqrt(λ[i])
    i = i + 1
print(Σ)