import numpy as np


s1 = '_' + input()
s2 = '_' + input()
if len(s1) >= len(s2):
    temp = s1
    s1 = s2
    s2 = temp
m = len(s1)
n = len(s2)
S = np.zeros([m, n])


def Weight(a, b):
    if a == b:
        return 2
    else:
        return -1


def get_Score(S, m, n, s1, s2):
    for i in range(1, m):
        for j in range(1, n):
            S[i][j] = max(0, S[i-1, j-1] + Weight(s1[i], s2[j]), S[i-1, j] + Weight(s1[i], ' '), S[i, j-1] + Weight(' ', s2[j]))
    return S


def find_max(S):
    x, y = np.where(S == np.max(S))
    return x, y


def add_Blank(S, m, n, s1, s2, i, j):
    t1 = ''
    t2 = ''
    for k in range(n-j-1):
        t1 = t1 + '_'
        t2 = t2 + s2[n-k-1]
    while i > 0 and j > 0 and S[i][j] != 0:
        Score = S[i][j]
        ScoreDiag = S[i-1][j-1]
        ScoreUp = S[i][j-1]
        ScoreLeft = S[i-1][j]
        if Score == ScoreDiag + Weight(s1[i], s2[j]):
            t1 = t1 + s1[i]
            t2 = t2 + s2[j]
            i = i - 1
            j = j - 1
        elif Score == ScoreLeft - 1:
            t1 = t1 + s1[i]
            t2 = t2 + '_'
            i = i - 1
        elif Score == ScoreUp - 1:
            t1 = t1 + '_'
            t2 = t2 + s2[j]
            j = j - 1
    for q in range(j):
        t1 = t1 + '_'
        t2 = t2 + s2[j-q]
    return t1[::-1], t2[::-1]

def local_alignment(S, m, n, s1, s2):
    get_Score(S, m, n, s1, s2)
    x, y = find_max(S)
    cnt = len(x)
    k = 1
    while cnt > 0 and k <= 10:
        i = x[cnt-1]
        j = y[cnt-1]
        r1, r2 = add_Blank(S, m, n, s1, s2, i, j)
        print(k)
        print(r1)
        print(r2)
        cnt = cnt - 1
        k = k + 1


local_alignment(S, m, n, s1, s2)