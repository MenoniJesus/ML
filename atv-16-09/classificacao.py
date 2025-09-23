import numpy as np
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

        for clf_name, classific in clfs.items():
            classific.fit(xtr, ytr)
            ypred = classific.predict(xte)
            f1 = round(metrics.f1_score(yte, ypred, average='macro'), 2)
            acc = round(metrics.accuracy_score(yte, ypred), 2)
            resultado = {
                'nome': 'tic-tac-toe',
                'classificador': clf_name,
                'f1': f1,
                'acc': acc
            }
            resultados_rodada.append(resultado)

        ytarg = list(ytarg[part:]) + list(ytarg[:part])
        xdata = list(xdata[part:]) + list(xdata[:part])

    return resultados_rodada

def salvar_resultados_customizados(todas_rodadas, nome_dataset='tic-tac-toe'):
    classificadores = ['perceptron', 'svm', 'bayes', 'trees', 'knn']
    metricas = ['f1', 'acc']
    linhas = []

    for clf in classificadores:
        for metrica in metricas:
            resultados = [
                rodada[metrica]
                for rodada in todas_rodadas
                if rodada['nome'] == nome_dataset and rodada['classificador'] == clf
            ]
            while len(resultados) < 20:
                resultados.append(None)
            linha = [nome_dataset, clf, metrica] + resultados[:20]
            linhas.append(linha)

    colunas = ['nome_dataset', 'nome_classificador', 'metrica'] + [f'resultado_{i+1}' for i in range(20)]
    df = pd.DataFrame(linhas, columns=colunas)
    df.to_csv('result.csv', index=False)
    print('Resultados salvos em result.csv')

todas_rodadas = []
for i in range(4):
    idx = np.arange(len(alltarg))
    np.random.shuffle(idx)
    xdata_shuffled = alldata[idx]
    ytarg_shuffled = alltarg[idx]
    todas_rodadas.extend(get_cv_value(xdata_shuffled, ytarg_shuffled))

salvar_resultados_customizados(todas_rodadas)
