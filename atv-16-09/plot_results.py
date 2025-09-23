import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')
metodos = ['perceptron', 'svm', 'bayes', 'trees', 'knn']

# Boxplot de acurácia
acuracias = []
for metodo in metodos:
    linha = df[(df['nome_classificador'] == metodo) & (df['metrica'] == 'acc')]
    resultados = linha.iloc[0, 3:23].values.astype(float)
    acuracias.append(resultados)
plt.boxplot(acuracias, tick_labels=metodos)
plt.xlabel('Algoritmo')
plt.ylabel('Acurácia')
plt.title('Boxplot de Acurácia por Algoritmo')
plt.tight_layout()
plt.show()

# Boxplot de F1
f1s = []
for metodo in metodos:
    linha = df[(df['nome_classificador'] == metodo) & (df['metrica'] == 'f1')]
    resultados = linha.iloc[0, 3:23].values.astype(float)
    f1s.append(resultados)
plt.boxplot(f1s, tick_labels=metodos)
plt.xlabel('Algoritmo')
plt.ylabel('F1-Measure')
plt.title('Boxplot de F1 por Algoritmo')
plt.tight_layout()
plt.show()
