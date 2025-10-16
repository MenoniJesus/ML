import pandas as pd
import numpy as np

print("Carregando dados originais...")
# Carrega o arquivo original (já limpo pelo trabalho.py)
df_original = pd.read_csv('/home/menoni/Documentos/Facul/ML/atv-30-09/all.csv')

print(f"Dados carregados: {len(df_original)} registros")
print(f"Colunas: {list(df_original.columns)}")

# Define as colunas de valores (v1 a v20)
value_cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
              'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20']

# ============================================================================
# ETAPA 1: MELT - Transforma colunas v1-v20 em linhas
# ============================================================================
print("\n=== ETAPA 1: FAZENDO MELT ===")

# Faz o melt: transforma as 20 colunas v1-v20 em linhas
df_melted = pd.melt(
    df_original,
    id_vars=['dataset', 'classifier', 'metric', 'author'],  # mantém essas colunas
    value_vars=value_cols,                                   # "derrete" essas colunas
    var_name='fold',                                         # nome da nova coluna (v1, v2, v3...)
    value_name='value'                                       # nome da coluna com os valores
)

print(f"Após melt: {len(df_melted)} registros")
print("Estrutura após melt:")
print("dataset | classifier | metric | author | fold | value")
print(df_melted.head())

# Verifica se há valores não numéricos
print(f"\nTipo da coluna value: {df_melted['value'].dtype}")

# Tenta converter para numérico e vê o que falha
df_melted['value_numeric'] = pd.to_numeric(df_melted['value'], errors='coerce')
problemas = df_melted[df_melted['value_numeric'].isna() & df_melted['value'].notna()]

if len(problemas) > 0:
    print(f"⚠️ Encontrados {len(problemas)} valores não numéricos:")
    print(problemas[['dataset', 'classifier', 'metric', 'author', 'fold', 'value']].head(10))
    print(f"Exemplos: {problemas['value'].unique()[:5]}")

# Limpa dados problemáticos
df_clean = df_melted.dropna(subset=['value_numeric']).copy()
df_clean['value'] = df_clean['value_numeric']
df_clean = df_clean.drop('value_numeric', axis=1)

print(f"Após limpeza: {len(df_clean)} registros (removidos: {len(df_melted) - len(df_clean)})")

# Salva o arquivo melted
melted_file = '/home/menoni/Documentos/Facul/ML/atv-30-09/all_melted.csv'
df_clean.to_csv(melted_file, index=False)
print(f"✅ Arquivo melted salvo: {melted_file}")

# ============================================================================
# ETAPA 2: PIVOT - Lê arquivo melted e gera pivot final
# ============================================================================
print("\n=== ETAPA 2: FAZENDO MÉDIA + PIVOT ===")

# Carrega o arquivo melted que acabou de ser gerado
print("Carregando arquivo melted...")
df_melted_loaded = pd.read_csv(melted_file)
print(f"Dados melted carregados: {len(df_melted_loaded)} registros")

# Calcula médias por grupo
print("Calculando médias por grupo...")
df_summary = df_melted_loaded.groupby(['dataset', 'classifier', 'metric', 'author'])['value'].mean().reset_index()
df_summary.rename(columns={'value': 'media'}, inplace=True)

print(f"Após calcular médias: {len(df_summary)} registros")
print(f"Métricas únicas: {sorted(df_summary['metric'].unique())}")

# Faz o pivot para transformar métricas em colunas
print("Fazendo pivot das métricas...")
df_pivot = df_summary.pivot_table(
    index=['dataset', 'classifier', 'author'], 
    columns='metric', 
    values='media',
    fill_value=np.nan
).reset_index()

# Remove o nome do índice das colunas
df_pivot.columns.name = None

print(f"Após pivot: {len(df_pivot)} registros")
print(f"Colunas finais: {list(df_pivot.columns)}")

# Salva o resultado final
output_file = '/home/menoni/Documentos/Facul/ML/atv-30-09/all_pivot.csv'
df_pivot.to_csv(output_file, index=False)
print(f"✅ Arquivo final salvo: {output_file}")

print("\n=== RESULTADO FINAL ===")
print(df_pivot.head())

print(f"\nEstatísticas:")
print(f"- Datasets: {len(df_pivot['dataset'].unique())}")
print(f"- Classificadores: {len(df_pivot['classifier'].unique())}")  
print(f"- Autores: {len(df_pivot['author'].unique())}")

# Mostra as colunas de métricas
metric_columns = [col for col in df_pivot.columns if col not in ['dataset', 'classifier', 'author']]
print(f"- Métricas: {metric_columns}")

# ============================================================================
# ETAPA EXTRA: ARQUIVO FILTRADO - Só SVM + F1 (versão melted)
# ============================================================================
print("\n=== ETAPA EXTRA: GERANDO ARQUIVO SVM F1 ONLY ===")

# Filtra apenas SVM + F1 do arquivo melted
svm_f1_melted = df_melted_loaded[
    (df_melted_loaded['classifier'] == 'svm') & 
    (df_melted_loaded['metric'] == 'f1')
].copy()

print(f"Dados filtrados SVM + F1: {len(svm_f1_melted)} registros")
print(f"Datasets únicos: {len(svm_f1_melted['dataset'].unique())}")
print(f"Autores únicos: {len(svm_f1_melted['author'].unique())}")

# Salva arquivo específico para SVM F1
svm_f1_file = '/home/menoni/Documentos/Facul/ML/atv-30-09/svm_f1_melted.csv'
svm_f1_melted.to_csv(svm_f1_file, index=False)
print(f"✅ Arquivo SVM F1 salvo: {svm_f1_file}")

print("\nPrimeiras linhas do arquivo SVM F1:")
print(svm_f1_melted.head())

print("\n=== RESUMO COMPLETO ===")
print("✅ ETAPA 1: MELT → all.csv → all_melted.csv")
print("✅ ETAPA 2: MÉDIA + PIVOT → all_melted.csv → all_pivot.csv")
print("✅ ETAPA EXTRA: FILTRO → all_melted.csv → svm_f1_melted.csv")
print(f"📊 Arquivos gerados:")
print(f"   - all_melted.csv: {len(df_melted_loaded)} registros (todos os dados)")
print(f"   - all_pivot.csv: {len(df_pivot)} registros (médias pivotadas)")
print(f"   - svm_f1_melted.csv: {len(svm_f1_melted)} registros (só SVM F1)")