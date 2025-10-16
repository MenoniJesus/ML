# Análise ROC/AUC - Classificadores Machine Learning

Este projeto implementa uma análise completa de curvas ROC (Receiver Operating Characteristic) e AUC (Area Under Curve) para 5 diferentes algoritmos de classificação no dataset Tic-Tac-Toe.

## 📊 Resultados Principais

| Classificador | AUC   | Acurácia | F1-Score | Performance |
|---------------|-------|----------|----------|-------------|
| **KNN**       | 0.900 | 0.849    | 0.814    | Excelente   |
| Decision Tree | 0.875 | 0.896    | 0.883    | Muito Bom   |
| SVM           | 0.827 | 0.682    | 0.484    | Muito Bom   |
| Naive Bayes   | 0.704 | 0.708    | 0.558    | Bom         |
| Perceptron    | 0.629 | 0.568    | 0.558    | Razoável    |

## 🎯 Objetivos

1. **Compreender Curvas ROC**: Análise da capacidade dos classificadores em distinguir entre classes
2. **Calcular AUC**: Métrica única que resume a performance da curva ROC
3. **Matriz de Confusão**: Visualização detalhada dos acertos e erros
4. **Comparação de Algoritmos**: Ranking e análise comparativa

## 📈 Interpretação dos Resultados

### Curva ROC
- **Eixo X**: Taxa de Falsos Positivos (FPR)
- **Eixo Y**: Taxa de Verdadeiros Positivos (TPR)
- **Ideal**: Curva próxima ao canto superior esquerdo
- **Baseline**: Linha diagonal representa classificador aleatório (AUC = 0.5)

### Escala AUC
- **0.9 - 1.0**: Excelente
- **0.8 - 0.9**: Muito Bom  
- **0.7 - 0.8**: Bom
- **0.6 - 0.7**: Razoável
- **0.5 - 0.6**: Ruim
- **0.5**: Aleatório

## 🔧 Como Executar

```bash
# 1. Executar análise básica
python classificacao.py

# 2. Executar análise completa com visualizações
python roc_analysis.py

# 3. Ver interpretação detalhada
python interpret_results.py
```

## 📁 Arquivos Gerados

### Visualizações
- `roc_curves_comparison.png` - Curvas ROC de todos os classificadores
- `confusion_matrix_[classificador].png` - Matriz de confusão para cada algoritmo

### Dados
- `results_summary.csv` - Tabela resumo com todas as métricas

## 🧠 Análise por Algoritmo

### 🥇 KNN (k=7) - CAMPEÃO
- **AUC**: 0.900 (Excelente)
- **Por que funciona bem**: Tic-tac-toe tem padrões locais bem definidos
- **Características**: Classificação baseada em vizinhança

### 🥈 Decision Tree - VICE-CAMPEÃO
- **AUC**: 0.875 (Muito Bom)
- **Por que funciona bem**: Excelente para modelar regras do jogo
- **Características**: Cria árvore de decisões hierárquica

### 🥉 SVM - TERCEIRO LUGAR
- **AUC**: 0.827 (Muito Bom)
- **Por que funciona bem**: Eficaz em encontrar fronteira ótima de separação
- **Características**: Maximiza margem entre classes

### 4️⃣ Naive Bayes
- **AUC**: 0.704 (Bom)
- **Limitação**: Assume independência entre features
- **Performance**: Razoável apesar das limitações

### 5️⃣ Perceptron
- **AUC**: 0.629 (Razoável)
- **Limitação**: Classificador linear simples
- **Performance**: Melhor que aleatório, mas limitado

## 📚 Conceitos Aprendidos

### ROC (Receiver Operating Characteristic)
- Gráfico que mostra performance em diferentes thresholds
- Originalmente desenvolvido para radares na 2ª Guerra Mundial
- Hoje usado extensivamente em ML para avaliação de classificadores

### Matriz de Confusão
```
                Predito
              Pos    Neg
Real Pos     TP     FN
     Neg     FP     TN
```

### Métricas Derivadas
- **TPR (Sensibilidade)**: TP/(TP+FN)
- **FPR**: FP/(FP+TN)  
- **Especificidade**: TN/(TN+FP)
- **Precisão**: TP/(TP+FP)

## 🎓 Conclusões

1. **KNN é o melhor classificador** para este dataset específico
2. **Todos os algoritmos** superaram performance aleatória
3. **Decision Tree** combina alta AUC com alta acurácia
4. **SVM** tem boa AUC mas acurácia mais baixa (mais conservador)
5. **Metodologia 80/20** foi adequada para avaliação robusta

## 🛠️ Dependências

```python
pandas
numpy
scikit-learn
matplotlib
scipy
```

## 👨‍🎓 Contexto Acadêmico

Este trabalho foi desenvolvido como parte dos estudos de Machine Learning, focando especificamente em:
- Avaliação de classificadores binários
- Análise ROC/AUC  
- Interpretação de resultados
- Comparação de algoritmos

---

**Autor**: Leandro Menoni  
**Disciplina**: Machine Learning  
**Dataset**: Tic-Tac-Toe Endgame  
**Data**: Outubro 2025