letras = 'abcdefghijklmnopqrstuvwxyz@#*+'

# criando uma lista de letras
lista = list(letras)
print('len(lista):', len(lista))
print(lista)

# importando a biblioteca matemática: numpy
import numpy as np

# transformando nossa lista em um array do numpy
lista = np.array(lista)

# dado a lista anterior, faça os exercícios:

meio = 15

# 1- capturar os primeiros 10 elementos e imprimir na tela
# 2- capturar os últimos 10 elementos e imprimir na tela
# 3- capturar os 10 elementos do meio e imprimir na tela
# 4- imprimir o 21o elemento apenas
print(lista[20])

# 5- imprimir todos elementos, menos os 5 últimos
print(lista[ : -5])

# 6- imprimir todos elementos do início até o meio
print(lista[ : meio])

# 7.1- imprimir todos elementos do meio até o final, em ordem reversa
print(lista[len : meio : -1])

# 8- imprimir todos elementos a partir do 5 , menos os 5 últimos
print(lista[5 : -5])

# 9- imprimir o 12 elemento
print(lista[12])

# 10- fazer um laço que repita 10 vezes, imprimindo cada vez 3 elementos
for i in range(10):
    ix = i*3
    print(lista[ ix : ix + 3])