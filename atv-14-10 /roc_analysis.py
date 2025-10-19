import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from dataset import data_set_v2, normalizacao_data

# Configurar matplotlib para salvar arquivos
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)

def load_dataset():
    import os
    # Obter o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'tic-tac-toe.csv')
    dataF = data_set_v2(csv_path)
    data2 = normalizacao_data(dataF['dados'])
    return data2

def plot_confusion_matrix(y_true, y_pred, classifier_name, save_path=None):
    """Plota e salva a matriz de confusão para um classificador"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão - {classifier_name}', fontsize=14)
    plt.colorbar()
    
    # Adiciona os números na matriz
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    
    # Obter labels das classes
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {save_path}")
    
    plt.close()

def plot_roc_curves(classifiers_results, save_path=None):
    """Plota e salva as curvas ROC de todos os classificadores"""
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, fpr, tpr, auc_score) in enumerate(classifiers_results):
        plt.plot(fpr, tpr, color=colors[i], lw=3, 
                label=f'{name} (AUC = {auc_score:.3f})')
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Classificador Aleatório (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
    plt.title('Curvas ROC - Comparação dos Classificadores\nDataset: Tic-Tac-Toe', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curvas ROC salvas em: {save_path}")
    
    plt.close()

def create_summary_table(results_summary, save_path=None):
    """Cria uma tabela resumo dos resultados"""
    df = pd.DataFrame(results_summary)
    df = df.round(3)
    df = df.sort_values('AUC', ascending=False)
    
    print("\n" + "="*60)
    print("RESUMO DOS RESULTADOS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Resumo salvo em: {save_path}")
    
    return df

def analyze_roc_performance(classifiers_results):
    """Analisa a performance dos classificadores"""
    print("\n" + "="*60)
    print("ANÁLISE DE PERFORMANCE ROC/AUC")
    print("="*60)
    
    # Ordenar por AUC
    sorted_results = sorted(classifiers_results, key=lambda x: x[3], reverse=True)
    
    for i, (name, fpr, tpr, auc_score) in enumerate(sorted_results, 1):
        print(f"{i}º lugar: {name}")
        print(f"   AUC: {auc_score:.3f}")
        
        if auc_score >= 0.9:
            performance = "Excelente"
        elif auc_score >= 0.8:
            performance = "Muito Bom"
        elif auc_score >= 0.7:
            performance = "Bom"
        elif auc_score >= 0.6:
            performance = "Razoável"
        else:
            performance = "Ruim"
        
        print(f"   Performance: {performance}")
        print(f"   Interpretação: ", end="")
        
        if auc_score > 0.8:
            print("Classificador muito eficaz na distinção entre classes")
        elif auc_score > 0.6:
            print("Classificador moderadamente eficaz")
        else:
            print("Classificador pouco eficaz, próximo ao aleatório")
        
        print()

def evaluate_classifiers_with_visualization():
    """Avalia todos os classificadores com visualizações salvas"""
    
    # Carrega o dataset
    data = load_dataset()
    X = data.drop(columns='class').values
    y = data['class'].values
    
    print("ANÁLISE ROC/AUC - DATASET TIC-TAC-TOE")
    print("="*50)
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Tamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")
    print(f"Número de classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    print("-" * 50)
    
    # Definir classificadores
    classifiers = {
        'Perceptron': Perceptron(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, gamma='auto', random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'KNN': KNeighborsClassifier(n_neighbors=7)
    }
    
    results_for_roc = []
    results_summary = []
    
    # Treinar e avaliar cada classificador
    for name, classifier in classifiers.items():
        print(f"\n=== {name} ===")
        
        # Treinar o classificador
        classifier.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = classifier.predict(X_test)
        
        # Verificar se o classificador tem predict_proba
        if hasattr(classifier, 'predict_proba'):
            y_pred_proba = classifier.predict_proba(X_test)
        else:
            # Para classificadores sem predict_proba (como Perceptron)
            if hasattr(classifier, 'decision_function'):
                decision_scores = classifier.decision_function(X_test)
                # Converter scores em probabilidades aproximadas usando sigmoid
                from scipy.special import expit
                if decision_scores.ndim == 1:
                    prob_pos = expit(decision_scores)
                    y_pred_proba = np.column_stack([1 - prob_pos, prob_pos])
                else:
                    y_pred_proba = expit(decision_scores)
                    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
            else:
                n_classes = len(np.unique(y))
                y_pred_proba = np.zeros((len(y_pred), n_classes))
                for i, pred in enumerate(y_pred):
                    y_pred_proba[i, pred] = 1.0
        
        # Métricas básicas
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        precision = metrics.precision_score(y_test, y_pred, average='macro')
        recall = metrics.recall_score(y_test, y_pred, average='macro')
        
        print(f"Acurácia: {accuracy:.3f}")
        print(f"Precisão: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Matriz de confusão
        cm_path = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(y_test, y_pred, name, cm_path)
        
        # Para ROC curve
        try:
            if len(np.unique(y)) == 2:
                # Caso binário
                fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = metrics.auc(fpr, tpr)
            else:
                # Caso multiclasse - usa macro average
                auc_score = metrics.roc_auc_score(y_test, y_pred_proba, 
                                                multi_class='ovr', average='macro')
                # Para plot, vamos usar a classe 0 vs resto
                y_test_binary = (y_test == 0).astype(int)
                y_pred_binary = y_pred_proba[:, 0]
                fpr, tpr, _ = metrics.roc_curve(y_test_binary, y_pred_binary)
        except Exception as e:
            print(f"Erro ao calcular ROC/AUC para {name}: {e}")
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
            auc_score = 0.5
        
        print(f"AUC: {auc_score:.3f}")
        
        # Adiciona aos resultados
        results_for_roc.append((name, fpr, tpr, auc_score))
        results_summary.append({
            'Classificador': name,
            'Acurácia': accuracy,
            'Precisão': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc_score
        })
        
        print("-" * 30)
    
    # Plotar todas as curvas ROC juntas
    roc_path = "roc_curves_comparison.png"
    plot_roc_curves(results_for_roc, roc_path)
    
    # Criar tabela resumo
    summary_path = "results_summary.csv"
    create_summary_table(results_summary, summary_path)
    
    # Análise de performance
    analyze_roc_performance(results_for_roc)
    
    return results_for_roc, results_summary

if __name__ == '__main__':
    results_roc, results_summary = evaluate_classifiers_with_visualization()