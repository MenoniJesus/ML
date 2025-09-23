import pandas as pd;
import numpy as np;

def data_set ( fname ):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv('adult.csv')
    #cols = data.columns
    #ultima = cols[-1]
    
    for linha in data:
        for coluna in linha:
            if coluna == "?":
                df = data.drop(labels=linha)
    
    #nome_orig = data[ultima]
    #cls_orig, classes, cls_cnt = np.unique(nome_orig, return_inverse=True, return_counts=True)
    
    #classes = data[ultima]
    #df = data.drop(columns=ultima)
        
    #result['dados'] = df
    #result['classes'] = classes
    #result['cls-orig'] = cls_orig
    #result['cls-count'] = cls_cnt
    
    return result

#FNAME = 'adult.csv'
#if __name__ == '__main__':
    #data = data_set(FNAME)
    #print('-'*40)
    
    #ncls = len( data['cls-orig'] )
    #print(f"1 - Quantidade de classes: {ncls}")    
    #print("2 - Numero de itens para cada classe: ", data['cls-count'])
    
    #print('-'*40)
    #soma = np.sum( data['cls-count'] )
    #result = []
    #for vlr in data['cls-count']:
    #    result.append(soma / vlr)
        
    #max = np.max( result )
    #print('Valor Maximo --> ', max)