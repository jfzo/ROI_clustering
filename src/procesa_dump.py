"""
Usado para generar datos CSV a partir de los resultados "escupidos" por el script que sintoniza los
parametros del SNN-clustering.
"""

import numpy as np
fin = open("dump_sintonizacion.txt")
process_next = False
current_row = 0
N = 0
for A in fin:
    if len(A.strip()) == 0:
        continue
    if A.lower().startswith('uups'):
        continue
    N += 1
N = N / 2
fin.seek(0)

results = np.zeros((N, 7))

for A in fin:
    if process_next:
        process_next = False
        Eps, MinPts, K, VM = map(float, [A.strip().split()[i] for i in [1,3,5,8]])
        print Eps, MinPts, K, VM
        results[current_row, 0] = K
        results[current_row, 1] = Eps
        results[current_row, 2] = MinPts
        results[current_row, 3] = VM
        current_row += 1
        continue
    if len(A.strip()) == 0:
        continue
    if A.lower().startswith('uups'):
        continue
    n_cp, _, n_np, n_cls = map(int, [A.strip().split()[i] for i in [1,3,5,7]])
    results[current_row, 4] = n_cp
    results[current_row, 5] = n_np
    results[current_row, 6] = n_cls
    process_next=True
fin.close()

