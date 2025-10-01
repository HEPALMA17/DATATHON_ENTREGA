"""
Módulo de avaliação de modelos para o Decision AI
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Classe para avaliação de modelos de machine learning
    """
    
    def __init__(self):
        """Inicializa o avaliador de modelos"""
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Avalia um modelo usando métricas padrão
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Labels de teste
            model_name: Nome do modelo para identificação
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        try:
            # Faz predições
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Tenta obter probabilidades se disponível
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calcula métricas
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Adiciona AUC se probabilidades estão disponíveis
            if y_pred_proba is not None:
                try:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    metrics['auc'] = None
            else:
                metrics['auc'] = None
            
            # Armazena resultados
            self.evaluation_results[model_name] = metrics
            
            logger.info(f"Modelo {model_name} avaliado com sucesso")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo {model_name}: {e}")
            return None
    
    def evaluate_multiple_models(self, models_dict, X_test, y_test):
        """
        Avalia múltiplos modelos
        
        Args:
            models_dict: Dicionário com {nome: modelo}
            X_test: Features de teste
            y_test: Labels de teste
            
        Returns:
            dict: Resultados de avaliação para todos os modelos
        """
        results = {}
        
        for name, model in models_dict.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            if result:
                results[name] = result
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='f1_weighted'):
        """
        Realiza validação cruzada
        
        Args:
            model: Modelo para avaliar
            X: Features
            y: Labels
            cv: Número de folds
            scoring: Métrica de scoring
            
        Returns:
            dict: Resultados da validação cruzada
        """
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            results = {
                'scores': scores,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'min_score': scores.min(),
                'max_score': scores.max()
            }
            
            logger.info(f"Validação cruzada - Média: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na validação cruzada: {e}")
            return None
    
    def compare_models(self, metrics_list):
        """
        Compara múltiplos modelos baseado em métricas
        
        Args:
            metrics_list: Lista de dicionários com métricas
            
        Returns:
            pd.DataFrame: DataFrame com comparação dos modelos
        """
        try:
            comparison_data = []
            
            for metrics in metrics_list:
                if metrics:
                    comparison_data.append({
                        'Modelo': metrics['model_name'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1'],
                        'AUC': metrics.get('auc', 'N/A')
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df = df.sort_values('F1-Score', ascending=False)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erro ao comparar modelos: {e}")
            return pd.DataFrame()
    
    def plot_confusion_matrix(self, confusion_matrix, model_name="Model", labels=None):
        """
        Plota matriz de confusão
        
        Args:
            confusion_matrix: Matriz de confusão
            model_name: Nome do modelo
            labels: Labels das classes
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels or ['Negativo', 'Positivo'],
                yticklabels=labels or ['Negativo', 'Positivo']
            )
            plt.title(f'Matriz de Confusão - {model_name}')
            plt.ylabel('Valor Real')
            plt.xlabel('Valor Predito')
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Erro ao plotar matriz de confusão: {e}")
            return None
    
    def generate_evaluation_report(self, model_name=None):
        """
        Gera relatório de avaliação
        
        Args:
            model_name: Nome do modelo específico (opcional)
            
        Returns:
            str: Relatório formatado
        """
        try:
            if model_name and model_name in self.evaluation_results:
                results = {model_name: self.evaluation_results[model_name]}
            else:
                results = self.evaluation_results
            
            if not results:
                return "Nenhum resultado de avaliação disponível."
            
            report_lines = ["=" * 60]
            report_lines.append("RELATÓRIO DE AVALIAÇÃO DE MODELOS")
            report_lines.append("=" * 60)
            
            for name, metrics in results.items():
                report_lines.append(f"\n🤖 MODELO: {name}")
                report_lines.append("-" * 40)
                report_lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
                report_lines.append(f"Precision: {metrics['precision']:.4f}")
                report_lines.append(f"Recall:    {metrics['recall']:.4f}")
                report_lines.append(f"F1-Score:  {metrics['f1']:.4f}")
                
                if metrics.get('auc'):
                    report_lines.append(f"AUC:       {metrics['auc']:.4f}")
                
                report_lines.append(f"\nMatrix de Confusão:")
                report_lines.append(str(metrics['confusion_matrix']))
            
            report_lines.append("\n" + "=" * 60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return f"Erro ao gerar relatório: {e}"
    
    def get_best_model(self, metric='f1'):
        """
        Retorna o melhor modelo baseado em uma métrica
        
        Args:
            metric: Métrica para comparação ('accuracy', 'precision', 'recall', 'f1', 'auc')
            
        Returns:
            dict: Informações do melhor modelo
        """
        try:
            if not self.evaluation_results:
                return None
            
            best_score = -1
            best_model = None
            
            for name, metrics in self.evaluation_results.items():
                score = metrics.get(metric, 0)
                if score and score > best_score:
                    best_score = score
                    best_model = {
                        'name': name,
                        'score': score,
                        'metrics': metrics
                    }
            
            return best_model
            
        except Exception as e:
            logger.error(f"Erro ao encontrar melhor modelo: {e}")
            return None
    
    def save_evaluation_results(self, filepath):
        """
        Salva resultados de avaliação em arquivo
        
        Args:
            filepath: Caminho para salvar o arquivo
        """
        try:
            import json
            
            # Converte numpy arrays para listas para serialização JSON
            serializable_results = {}
            for name, metrics in self.evaluation_results.items():
                serializable_metrics = metrics.copy()
                
                # Converte matriz de confusão para lista
                if 'confusion_matrix' in serializable_metrics:
                    serializable_metrics['confusion_matrix'] = serializable_metrics['confusion_matrix'].tolist()
                
                serializable_results[name] = serializable_metrics
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Resultados de avaliação salvos em: {filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
    
    def load_evaluation_results(self, filepath):
        """
        Carrega resultados de avaliação de arquivo
        
        Args:
            filepath: Caminho do arquivo para carregar
        """
        try:
            import json
            
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            # Converte listas de volta para numpy arrays
            for name, metrics in loaded_results.items():
                if 'confusion_matrix' in metrics:
                    metrics['confusion_matrix'] = np.array(metrics['confusion_matrix'])
            
            self.evaluation_results = loaded_results
            logger.info(f"Resultados de avaliação carregados de: {filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar resultados: {e}")
