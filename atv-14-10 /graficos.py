import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results_summary.csv')

# Gráfico de barras - Acurácia
plt.figure(figsize=(8,5))
plt.bar(df['Classificador'], df['Acurácia'], color='skyblue')  # <-- com acento!
plt.title('Acurácia dos Classificadores')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.close()

# Gráfico de barras - F1-Score
plt.figure(figsize=(8,5))
plt.bar(df['Classificador'], df['F1-Score'], color='lightgreen')
plt.title('F1-Score dos Classificadores')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('f1score_comparison.png')
plt.close()