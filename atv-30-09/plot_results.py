import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=== ANÁLISE F1 DO SVM POR DATASET (com médias) ===")

# Carrega os dados individuais
df_all = pd.read_csv('/home/menoni/Documentos/Facul/ML/atv-30-09/all.csv')
svm_f1_data = df_all[(df_all['classifier'] == 'svm') & (df_all['metric'] == 'f1')]

# Carrega as médias
df_summary = pd.read_csv('/home/menoni/Documentos/Facul/ML/atv-30-09/all_pivot.csv')
svm_means = df_summary[df_summary['classifier'] == 'svm'].dropna(subset=['f1'])

print(f"Datasets com dados individuais: {len(svm_f1_data['dataset'].unique())}")
print(f"Datasets com médias: {len(svm_means['dataset'].unique())}")

# Prepara dados para boxplot por dataset
datasets_unicos = sorted(svm_f1_data['dataset'].unique())
f1_por_dataset = []
labels_dataset = []
medias_por_dataset = []

for dataset in datasets_unicos:
    # Dados individuais para boxplot
    dataset_data = svm_f1_data[svm_f1_data['dataset'] == dataset]
    valores_dataset = []
    for _, row in dataset_data.iterrows():
        valores = row[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
                      'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20']].values
        valores_limpos = [float(v) for v in valores if pd.notna(v)]
        valores_dataset.extend(valores_limpos)
    
    # Média do dataset (do melt_pivot.csv)
    dataset_means = svm_means[svm_means['dataset'] == dataset]['f1']
    media_dataset = dataset_means.mean() if len(dataset_means) > 0 else np.nan
    
    if valores_dataset:  # Só adiciona se há dados
        f1_por_dataset.append(valores_dataset)
        labels_dataset.append(dataset)
        medias_por_dataset.append(media_dataset)

# Cria o gráfico
plt.figure(figsize=(16, 8))

# Boxplot dos valores individuais
bp = plt.boxplot(f1_por_dataset, tick_labels=labels_dataset, patch_artist=True)

# Customiza cores dos boxplots
colors = ['lightblue' if not np.isnan(media) else 'lightgray' for media in medias_por_dataset]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Sobrepõe as médias como pontos vermelhos (menores)
x_positions = range(1, len(labels_dataset) + 1)
valid_means = [(i+1, media) for i, media in enumerate(medias_por_dataset) if not np.isnan(media)]

if valid_means:
    x_valid, y_valid = zip(*valid_means)
    plt.scatter(x_valid, y_valid, color='red', s=50, marker='D', 
               zorder=5, edgecolors='darkred', linewidth=1)

# Configurações do gráfico
plt.xlabel('Dataset')
plt.ylabel('F1')
plt.title('Distribuição F1 do SVM por Dataset (com Médias)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Sem caixa de estatísticas para manter o gráfico limpo

plt.tight_layout()
plt.show()

print("\nAnálise completa! 📊")
print(f"Gráfico mostra {len(labels_dataset)} datasets com boxplots dos valores individuais")
print("Pontos vermelhos (losangos) representam as médias por dataset do arquivo melt_pivot.csv")
