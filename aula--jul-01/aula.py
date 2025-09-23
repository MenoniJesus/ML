from dataset import data_set
import pandas as pd


FNAME = 'datasets/adult/adult.csv'

if __name__ == '__main__':
    data = data_set(FNAME)
    for key, value in data.items():
        print(key)

    fname = FNAME.split('/')
    fname = fname[-1]
    print('fname -->', fname)
    
    # todo: salvar adult--dados.csv
    with open("dados.csv", "w") as arquivo:
        arquivo.write(data['dados'].to_csv(index=False))
     
    # todo: salvar adult--classes.csv
    pd.DataFrame(data['classes'], columns=['class']).to_csv("classes.csv", index=False)
    
    # ok: remover numeros invalidos...
    
    Xtreino = None
    Xteste = None
    Ytreino = None
    Yteste = None
    