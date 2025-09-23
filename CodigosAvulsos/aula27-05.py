import numpy as np

lista = [0,1,2,3,4,5,6,7,8,9]
lista = np.array(lista)

print(lista)
lista = lista.reshape( -1, 2 )
print(lista)
print(lista.T)

tabela = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z', '@', '#', '*', '+']

tabela = np.array(tabela)
tabela = tabela.reshape(-1, 10)
print(tabela)
tabela_t = tabela.T
print(tabela_t)

tabela2 = lista.T
for linha in tabela2:
    print('linha -->', linha)
    
for coluna in tabela2.T:
    print('coluna -->', coluna)
    
lista_nv = tabela.reshape(-1, 5)
print('--------------------------')
print(lista_nv)
print('--------------------------')
tabela3 = lista_nv[1: -1, 1: -1]
print(tabela3)

#.tolist()
#.flatten()

#lsita3 = [item for sublist in lista3 for item in sublist]