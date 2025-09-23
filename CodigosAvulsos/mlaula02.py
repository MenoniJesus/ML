import pandas as pd

pedrada = pd.read_csv('pedrinhas.csv')

lista_d = list(pedrada['desceu'])
lista_c = list(pedrada['classificacao'])
contagem = [0, 0]

for des,cls in zip(lista_d, lista_c):
    pass 