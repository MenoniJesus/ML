import pandas as pd

arquivo = pd.read_csv('pedrinhas.csv')

classificacao = arquivo['classificacao']
pedrinha = arquivo['desceu']
distancia = arquivo['distancia']
somaDes = {'joao': 0, 'maria': 0}
somaDis = {'joao': 0, 'maria': 0}

conta = {'joao': 0, 'maria': 0}

for cls, dsc, dis in zip(classificacao, pedrinha, distancia):
    if cls == 1: key = 'joao'
    if cls == 2: key = 'maria'
    somaDes[key] = somaDes[key] + dsc
    somaDis[key] = somaDis[key] + dis
    conta[key] += 1
    
mediaDes = {'joao': 0, 'maria': 0}
mediaDis = {'joao': 0, 'maria': 0}

mediaDes['joao'] = somaDes['joao'] / conta['joao']
mediaDes['maria'] = somaDes['maria'] / conta['maria']
mediaDis['joao'] = somaDis['joao'] / conta['joao']
mediaDis['maria'] = somaDis['maria'] / conta['maria']

print("Media da Descidas", mediaDes)
print("Meida da Distancia", mediaDis)