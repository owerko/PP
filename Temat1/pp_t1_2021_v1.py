import numpy as np
import itertools


def read_epoch(filename, n):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, skiprows=39, usecols=(n, n+1))
    return data


def reduce_idx_stable(idx_stable, q0):
    pairs = list(itertools.combinations(idx_stable, 2))
    # print(pairs)

    idx_pairs_stable = [i for i in pairs if q0[i] <= 1.5]
    idx_pairs_unstable = [i for i in pairs if q0[i] > 1.5]

    if len(idx_pairs_unstable)==0:
        return idx_stable

    else:
        # print("Pairs stable: ", idx_pairs_stable)         # wypisz pary stabilne
        # print("Pairs unstable: ", idx_pairs_unstable)     # wypisz pary niestabilne
        #
        # for i in idx_pairs_unstable:    # wypisz indeksy reperow niestabilnych
        #     print(i[0])
        #     print(i[1])

        unstables_count = np.zeros((n))
        for i in idx_pairs_unstable:
            unstables_count[i[0]] += 1
            unstables_count[i[1]] += 1

        # print("unstables count: ", unstables_count)     # ile razy repery wystepuja wsrod par niestabilnych

        un_max = sorted(range(len(unstables_count)), key=lambda k: unstables_count[k], reverse=True)
                        # posortowane repery od najgorszych
        print(" numer reperu do usuniecia:", un_max[0])
        idx_stable.pop(idx_stable.index(un_max[0]))

        return reduce_idx_stable(idx_stable, q0)        # rekurencja


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})        # dla celow podgladu

n = 9               # liczba reperow w sieci

filename1 = input('Podaj nazwe pliku z danymi (z1.txt): ')
filename2 = input('Podaj nazwe pliku z danymi (z2.txt): ')
# filename1 = "z6"
# filename2 = "z16"
data0 = read_epoch(filename1, 1)
data1 = read_epoch(filename1, 5)
data2 = read_epoch(filename2, 5)

data = np.concatenate((data0, data1, data2), axis=1)    # tabela jak w konspekcie
d = np.zeros((data.shape[0],data.shape[1]+3))           # dodanie 3 kolumn: dh, sdh, q
d[:, :-3] = data

d[:, 6] = 1000 * (d[:, 2] - d[:, 4])                            # dh [mm]
d[:, 7] = np.sqrt(np.power(d[:, 3], 2) + np.power(d[:, 5], 2))  # sdh [mm]
d[:, 8] = abs(d[:, 6]/d[:, 7])                                  # q
# print(d)                                                # wypisanie calej tabeli
q = d[:, 8]
# print(q)                                                # wypisanie wartosci q kolejno

q0 = np.zeros((n,n))
i0 = 0                              # wypisanie wartosci q w postaci macierzy (n, n)
for i in range(n):                  # q0 to macierz (n, n) zawierajaca wartosci q dla par
    i0 += i+1
    for j in range(i+1,n):
        q0[i, j] = q[n*i+j-i0]
q0 += np.transpose(q0)
print(q0)

sumq = np.sum(q0, axis=0)                       # suma wartosci q dla kazdego repera (kolumny/wiersza)
print("\nSuma q dla kazdego repera: ", sumq)                    # sumq min wskazuje na najlepszy reper

idx_min = sorted(range(len(sumq)), key=lambda k: sumq[k])       # sortowanie wg sumq
print("Repery od najlepszego (wg min(Suma q)): ", idx_min)

qs = q0[idx_min[0]]                             # q selected: wartosci q dla wybranego repera
print("Wiersz macierzy qs dla najlepszego repera: ", qs)
idx_stable = [i for i in range(len(qs)) if qs[i] <= 1.5]
print("\nWstepny wybor grupy: ", idx_stable)

print("Wyniki sprawdzenia grupy: ")
print(reduce_idx_stable(idx_stable, q0))        # jesli pozostale pary we wstepnie wybranej grupie maja q > 1.5
                                                # to zostana usuniete

# c.d. - do sprawdzenia grupy wspolne dla kolejnych reperow z listy idx_min




# https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
# https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
# https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
# https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
# https://stackoverflow.com/questions/7270321/finding-the-index-of-elements-based-on-a-condition-using-python-list-comprehensi/


