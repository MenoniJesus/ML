import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')
metodos = ['perceptron', 'svm', 'bayes', 'trees', 'knn']

# Gráfico de acurácia
acuracias = [df[f'{m}_acc'].values for m in metodos]
plt.boxplot(acuracias, tick_labels=metodos)
plt.xlabel('Algoritmo')
plt.ylabel('Acurácia')
plt.title('Boxplot de Acurácia por Algoritmo')
plt.tight_layout()
plt.show()

# Gráfico de F1
f1s = [df[f'{m}_f1'].values for m in metodos]
plt.boxplot(f1s, tick_labels=metodos)
plt.xlabel('Algoritmo')
plt.ylabel('F1-Measure')
plt.title('Boxplot de F1 por Algoritmo')
plt.tight_layout()
plt.show()
