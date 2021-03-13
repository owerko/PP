import numpy as np

n = 9

def read_epoch(filename, n):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, skiprows=39, usecols=(n, n + 1))
    return data

pary = read_epoch('z1.txt', 1)
pierwotna = read_epoch('z1.txt', 5)
wtorna = read_epoch('z2.txt', 5)

data = np.concatenate((pary, pierwotna, wtorna), axis=1)
d = np.zeros((data.shape[0], data.shape[1] + 3))
d[:, :-3] = data
d[:, 6] = 1000 * (d[:, 2] - d[:, 4])
d[:, 7] = np.sqrt(np.power(d[:, 3], 2) + np.power(d[:, 5], 2))
d[:, 8] = abs(d[:, 6] / d[:, 7])
q = d[:, 8]

q0 = np.zeros((n, n))
i0 = 0
for i in range(n):
    i0 += i + 1
    for j in range(i + 1, n):
        if q[n * i + j - i0] > 2:
            q0[i, j] = 1
        else:
            q0[i, j] = 0
q0 += np.transpose(q0)
N = {i: set(num for num, j in enumerate(row) if j) for i, row in enumerate(q0)}


def BronKerbosch(P, R=None, X=None):
    P = set(P)
    R = set() if R is None else R
    X = set() if X is None else X
    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from BronKerbosch(P=P.intersection(N[v]), R=R.union([v]), X=X.intersection(N[v]))
        X.add(v)


P = N.keys()

lista_stalych = [list(element) for element in list(BronKerbosch(P))]
liczba_stalych = [len(element) for element in lista_stalych]
m = max(liczba_stalych)
index = [i for i, j in enumerate(liczba_stalych) if j == m]

for i in index:
    print(f'Lista reperów srtałych - indeksy od 0: {lista_stalych[i]}')
