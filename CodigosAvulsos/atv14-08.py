from random import shuffle
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dataset import data_set, remover_dados_faltantes, _transform_col

def capturar_dados():
    data = data_set('tic-tac-toe.csv')
    data['dados'] = remover_dados_faltantes(data['dados'])  # Pré-processamento
    return data

def transformar_dados(data):
    # Transforma todas as colunas categóricas em numéricas
    for col in data.columns:
        transformed = _transform_col(data[col])
        data[col] = transformed['values']
    return data

data = capturar_dados()
xdata = transformar_dados(data['dados'])  # Transforma os dados em numéricos
ytarg = np.array(data['classes'])

xdata = np.array(xdata)

nums = list(range(len(ytarg)))
shuffle(nums)

xdata = xdata[nums]
ytarg = ytarg[nums]

size = len(ytarg)
particao = int(size * 0.6)  # treino -> 60%

xtreino = xdata[:particao]
ytreino = ytarg[:particao]

xteste = xdata[particao:]
yteste = ytarg[particao:]

perceptron = Perceptron(max_iter=100, random_state=42)
perceptron.fit(xtreino, ytreino)
yhat = perceptron.predict(xteste)

score = metrics.accuracy_score(yteste, yhat)
matrix = metrics.confusion_matrix(yteste, yhat)

print('Evaluating DS techniques:')
print('perceptron-score:', score)
print('confusion-matrix:\n', matrix)