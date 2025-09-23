import numpy as np

dados = np.genfromtxt('pedrinhas.csv', delimiter=',')

cont1 = 0
cont2 = 0
bigJ = 0
bigM = 0.0

for jogadas in dados:
    if jogadas[2] == 1:
        cont1+=1
        if bigJ < jogadas[0]:
            bigJ = jogadas[0]
            
    if jogadas[2] == 2:
        cont2+=1
        if bigM < jogadas[1]:
            bigM = jogadas[1]

print(f"Joao teve {cont1} jogadas")
print(f"A jogada mais longe de Joao foi {bigJ}")

print(f"Maria teve {cont2} jogadas")
print(f"A jogada mais funda de maria foi {bigM}")