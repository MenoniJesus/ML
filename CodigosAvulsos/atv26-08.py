import numpy as np
from random import shuffle
from sklearn import metrics

from sklearn.datasets import fetch_openml

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

from dataset import remover_dados_faltantes, _transform_col, data_set


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
xdata = transformar_dados(data['dados'])
ytarg = np.array(data['classes'])

xdata = np.array(xdata)


# embaralhar os dados
idx = list(range(len(ytarg)))
shuffle(idx)
part = int(len(ytarg)*0.8) # assumindo 75% treino

# xtr --> x_treino  ;  xte --> x_teste
xtr = xdata[ :part ]
ytr = ytarg[ :part ]
xte = xdata[ part: ]
yte = ytarg[ part: ]


rng = np.random.RandomState()

perceptron = Perceptron(max_iter=100,random_state=rng)
model_svc = SVC(probability=True, gamma='auto',random_state=rng)
model_bayes = GaussianNB()
model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
model_knn = KNeighborsClassifier(n_neighbors=7)

# colocando todos classificadores criados em um dicionario
clfs = {    'perceptron':   perceptron,
            'svm':          model_svc,
            'bayes':        model_bayes,
            'trees':        model_tree,
            'knn':          model_knn
        }

ytrue = yte
print('Treinando cada classificador e encontrando o score')

# Dicionário para armazenar os resultados de cada classificador
results = defaultdict(list)

# Número de repetições
n_repeats = 3

for _ in range(n_repeats):
    # Embaralhar os dados novamente para cada repetição
    shuffle(idx)
    xtr = xdata[:part]
    ytr = ytarg[:part]
    xte = xdata[part:]
    yte = ytarg[part:]

    for clf_name, classific in clfs.items():
        classific.fit(xtr, ytr)
        ypred = classific.predict(xte)
        f1 = metrics.f1_score(yte, ypred, average='macro')
        results[clf_name].append(f1)

# Calcular e exibir a média dos resultados
for clf_name, scores in results.items():
    mean_f1 = np.mean(scores)
    print(f'{clf_name} -- Media do f1-score apos {n_repeats} execucoes: {mean_f1:.4f}')