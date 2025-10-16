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

def load_dataset():
    dataF = data_set_v2('tic-tac-toe.csv')
    data2 = normalizacao_data(dataF['dados'])
    return data2

def load_dataset():
    dataF = data_set_v2('tic-tac-toe.csv')
    data2 = normalizacao_data(dataF['dados'])
    return data2

def plot_confusion_matrix(y_true, y_pred, classifier_name):
    """Plota a matriz de confusão para um classificador"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão - {classifier_name}')
    plt.colorbar()
    
    # Adiciona os números na matriz
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.show()

def plot_roc_curves(classifiers_results):
    """Plota as curvas ROC de todos os classificadores em um único gráfico"""
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, fpr, tpr, auc_score) in enumerate(classifiers_results):
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {auc_score:.3f})')
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Classificador Aleatório (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curvas ROC - Comparação dos Classificadores')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_classifiers():
    """Avalia todos os classificadores e gera ROC/AUC"""
    
    # Carrega o dataset
    data = load_dataset()
    X = data.drop(columns='class').values
    y = data['class'].values
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Tamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")
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
            # Usar decision_function se disponível
            if hasattr(classifier, 'decision_function'):
                decision_scores = classifier.decision_function(X_test)
                # Converter scores em probabilidades aproximadas usando sigmoid
                from scipy.special import expit
                if decision_scores.ndim == 1:
                    # Binário
                    prob_pos = expit(decision_scores)
                    y_pred_proba = np.column_stack([1 - prob_pos, prob_pos])
                else:
                    # Multiclasse
                    y_pred_proba = expit(decision_scores)
                    # Normalizar para que cada linha some 1
                    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
            else:
                # Último recurso: usar predições como probabilidades "duras"
                n_classes = len(np.unique(y))
                y_pred_proba = np.zeros((len(y_pred), n_classes))
                for i, pred in enumerate(y_pred):
                    y_pred_proba[i, pred] = 1.0
        
        # Métricas básicas
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        
        print(f"Acurácia: {accuracy:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Matriz de confusão
        plot_confusion_matrix(y_test, y_pred, name)
        
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
                # Para plot, vamos usar a classe positiva como referência
                # Transformando em problema binário (classe 0 vs resto)
                y_test_binary = (y_test == 0).astype(int)
                y_pred_binary = y_pred_proba[:, 0]
                fpr, tpr, _ = metrics.roc_curve(y_test_binary, y_pred_binary)
        except Exception as e:
            print(f"Erro ao calcular ROC/AUC para {name}: {e}")
            # Valores padrão em caso de erro
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
            auc_score = 0.5
        
        print(f"AUC: {auc_score:.3f}")
        
        # Adiciona aos resultados para plot ROC
        results_for_roc.append((name, fpr, tpr, auc_score))
        
        print("-" * 30)
    
    # Plotar todas as curvas ROC juntas
    plot_roc_curves(results_for_roc)
    
    return results_for_roc

if __name__ == '__main__':
    results = evaluate_classifiers()
