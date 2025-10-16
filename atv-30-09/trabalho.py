import os
import pandas as pd


BASEDIR='/home/menoni/Documentos/Facul/ML/atv-30-09/trabalhos'
RESULT='/home/menoni/Documentos/Facul/ML/atv-30-09/all.csv'

MYCOLS=['dataset', 'classifier', 'metric',
        'v1', 'v2', 'v3', 'v4',
        'v5', 'v6', 'v7', 'v8',
        'v9', 'v10', 'v11', 'v12',
        'v13', 'v14', 'v15', 'v16',
        'v17', 'v18', 'v19', 'v20',
        'author'
    ]

# Dicionários para padronização da nomenclatura
CLASSIFIER_MAPPING = {
    'perceptron': 'perceptron',
    'Perceptron': 'perceptron',
    ' perceptron': 'perceptron',
    'svm': 'svm',
    'SVM': 'svm',
    ' svm': 'svm',
    'SVC': 'svm',
    'bayes': 'naive_bayes',
    'Naive Bayes': 'naive_bayes',
    'NaiveBayes': 'naive_bayes',
    ' bayes': 'naive_bayes',
    'GaussianNB': 'naive_bayes',
    'trees': 'decision_tree',
    'Decision Tree': 'decision_tree',
    'DecisionTree': 'decision_tree',
    ' trees': 'decision_tree',
    'RandomForest': 'random_forest',
    'knn': 'knn',
    'KNN': 'knn',
    ' knn': 'knn',
    'KNeighbors': 'knn',
    'LogisticRegression': 'logistic_regression'
}

METRIC_MAPPING = {
    'f1-score': 'f1',
    'f1': 'f1',
    'F1-Score': 'f1',
    'F1': 'f1',
    'f1_score': 'f1',
    ' f1': 'f1',
    'F1-Measure': 'f1',
    'F1_Score': 'f1',
    'accuracy': 'acc',
    'Acurácia': 'acc',
    'ACC': 'acc',
    'acc': 'acc',
    ' acc': 'acc',
    'Accuracy': 'acc',
    'Acc': 'acc',
    'Acuracia': 'acc'
}

DATASET_MAPPING = {
    'tic-tac-toe': 'tic_tac_toe',
    'glass_identification': 'glass_identification',
    'winequality-red': 'wine_quality_red',
    'column_2C.csv': 'column_2c',
    'spambase': 'spambase',
    'insurance': 'insurance',
    'bank': 'bank'
}

def standardize_names(df):
    """Padroniza os nomes de classificadores, métricas e datasets"""
    # Padronizar classificadores
    df['classifier'] = df['classifier'].map(CLASSIFIER_MAPPING).fillna(df['classifier'])
    
    # Padronizar métricas
    df['metric'] = df['metric'].map(METRIC_MAPPING).fillna(df['metric'])
    
    # Padronizar datasets
    df['dataset'] = df['dataset'].map(DATASET_MAPPING).fillna(df['dataset'])
    
    return df

mylist = os.listdir(BASEDIR)
result = []
for fname in mylist:
    # se arquivo nao for csv, pula
    if fname[-3:] != 'csv': continue

    print(f"Processando: {fname}")
    df = pd.read_csv(BASEDIR+'/'+fname)
    df['author'] = fname[ :-4 ]
    df.columns = MYCOLS
    
    # Aplicar padronização
    df = standardize_names(df)
    
    result.append(df)

result = pd.concat(result, axis=0, ignore_index=True)
result.to_csv(RESULT, index=False)

print(f"\nArquivo {RESULT} gerado com sucesso!")
print(f"Total de registros: {len(result)}")
print(f"\nClassificadores únicos: {sorted(result['classifier'].unique())}")
print(f"\nMétricas únicas: {sorted(result['metric'].unique())}")
print(f"\nDatasets únicos: {sorted(result['dataset'].unique())}")
print(f"\nAutores únicos: {len(result['author'].unique())}")



# ------------------ abaixo, valores de CLASSIFICADOR desregulado
# [
#   'perceptron' 'svm' 'bayes' 'trees' 'knn' 'Perceptron' 'SVM'
#   'NaiveBayes' 'KNN' 'DecisionTree' 'Naive Bayes' 'Decision Tree'
#   ' perceptron' ' svm' ' bayes' ' trees' ' knn' 'GaussianNB' 'SVC'
#   'LogisticRegression' 'RandomForest' 'KNeighbors'
# ]


# ------------------ abaixo, valores de METRICAS desregulado
# [
#   'f1-score' 'accuracy' 'f1' 'F1-Score' 'Acurácia' 'ACC' 'F1'
#   'f1_score' 'acc' ' f1' ' acc' 'F1-Measure' 'Accuracy' 'Acc'
#   'F1_Score' 'Acuracia'
# ]