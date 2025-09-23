lista = [1, 2, 3, 3, 2, 1, 2, 3, 1, 2]
lista2 = ["mar", "maca", "banana", "mar", "aula", "aula"]

def contagem(_lista):
    resp = {}
    for item in _lista:
        if item not in resp: resp[item] = 0
        resp[item] += 1
        
    print("result ->", resp)
    
    
#contagem(lista)
#contagem(lista2)
print("itens totais --> ",z )