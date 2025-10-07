import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carrega o arquivo com dados padronizados
df = pd.read_csv('/home/menoni/Documentos/Facul/ML/atv-30-09/all.csv')

# Filtra apenas resultados do SVM com métrica F1
svm_f1_data = df[(df['classifier'] == 'svm') & (df['metric'] == 'f1')]

print(f"Total de registros SVM + F1: {len(svm_f1_data)}")
print(f"Datasets encontrados: {sorted(svm_f1_data['dataset'].unique())}")
print(f"Autores encontrados: {len(svm_f1_data['author'].unique())}")

# Coleta todos os valores de F1 do SVM
all_f1_values = []
for _, row in svm_f1_data.iterrows():
    # Pega os valores das colunas v1 a v20
    valores = row[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
                   'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20']].values
    # Remove valores NaN e converte para float
    valores_limpos = [float(v) for v in valores if pd.notna(v)]
    all_f1_values.extend(valores_limpos)

# Criar boxplot dos resultados F1 do SVM
plt.figure(figsize=(8, 6))
plt.boxplot([all_f1_values], tick_labels=['SVM'])
plt.xlabel('Classificador')
plt.ylabel('F1')
plt.title('Distribuição dos Resultados F1 para SVM')
plt.grid(True, alpha=0.3)

# Adicionar estatísticas no gráfico
stats_text = f'N = {len(all_f1_values)}\n'
stats_text += f'Média: {np.mean(all_f1_values):.3f}\n'
stats_text += f'Mediana: {np.median(all_f1_values):.3f}\n'
stats_text += f'Desvio: {np.std(all_f1_values):.3f}'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Opcional: Boxplot por dataset
datasets_unicos = sorted(svm_f1_data['dataset'].unique())
if len(datasets_unicos) > 1:
    plt.figure(figsize=(12, 6))
    f1_por_dataset = []
    labels_dataset = []
    
    for dataset in datasets_unicos:
        dataset_data = svm_f1_data[svm_f1_data['dataset'] == dataset]
        valores_dataset = []
        for _, row in dataset_data.iterrows():
            valores = row[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
                          'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20']].values
            valores_limpos = [float(v) for v in valores if pd.notna(v)]
            valores_dataset.extend(valores_limpos)
        
        if valores_dataset:  # Só adiciona se há dados
            f1_por_dataset.append(valores_dataset)
            labels_dataset.append(dataset)
    
    plt.boxplot(f1_por_dataset, tick_labels=labels_dataset)
    plt.xlabel('Dataset')
    plt.ylabel('F1')
    plt.title('Distribuição F1 do SVM por Dataset')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
