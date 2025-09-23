import numpy as np

tab = [[1,2,3],[4,5,6],[7,8,9]]

tab = np.array(tab)

print('ndim -->', tab.ndim)

print('mostrando indice --> ', tab[ 1 ,1 ])

print('mostrando indice --> ', tab[ -1 ])

print('coluna meio --> ', tab[ : ,1 ])

tab_t = [[]]

lista = [1,2,3,4,5,6,7,8,9]
lista = np.array(lista)

print(lista)
lista = lista.reshape( 3 ,3 )
print(lista)

