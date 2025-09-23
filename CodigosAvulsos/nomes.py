import json
from teste1 import contagem

with open ("pessoas.json", "r") as arquivo:
    dados = json.load(arquivo)
    

contagem(dados["nomes-pessoas"])