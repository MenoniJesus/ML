import numpy as np
from random import shuffle
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset import data_set_v2, normalizacao_data

def load_dataset():
    dataF = data_set_v2('tic-tac-toe.csv')
    data2 = normalizacao_data(dataF['dados'])
    return data2

data = load_dataset()
alldata = data.drop(columns='class').values
alltarg = data['class'].values

rng = np.random.RandomState()

def get_cv_value(xdata, ytarg):
    part = int(len(ytarg)*0.8)
    resultados_rodada = []
    for crossv in range(5):
        xtr = xdata[:part]
        ytr = ytarg[:part]
        xte = xdata[part:]
        yte = ytarg[part:]

        clfs = {
            'perceptron': Perceptron(max_iter=100, random_state=rng),
            'svm': SVC(probability=True, gamma='auto', random_state=rng),
            'bayes': GaussianNB(),
            'trees': DecisionTreeClassifier(random_state=rng, max_depth=10),
            'knn': KNeighborsClassifier(n_neighbors=7)
        }

        resultado = {}
        resultado['nome'] = 'tic-tac-toe'
        for clf_name, classific in clfs.items():
            classific.fit(xtr, ytr)
            ypred = classific.predict(xte)
            f1 = round(metrics.f1_score(yte, ypred, average='macro'), 2)
            acc = round(metrics.accuracy_score(yte, ypred), 2)
            resultado[f'{clf_name}'] = clf_name
            resultado[f'metrica_f1'] = f1
            resultado[f'{clf_name}_acc'] = acc

        resultados_rodada.append(resultado)

        ytarg = list(ytarg[part:]) + list(ytarg[:part])
        xdata = list(xdata[part:]) + list(xdata[:part])

    return resultados_rodada

def principal():
    todas_rodadas = []
    for exec_id in range(4):
        idx = list(range(len(alltarg)))
        shuffle(idx)
        xdata = alldata[idx]
        ytarg = alltarg[idx]
        rodadas = get_cv_value(xdata, ytarg)
        todas_rodadas.extend(rodadas)

    df = pd.DataFrame(todas_rodadas)
    df.to_csv('result.csv', index=False)
    print('Resultados salvos em result.csv')

principal()
