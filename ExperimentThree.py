# Importaciones necesarias
from automatic_labeling02 import *
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # o import joblib si tienes la versión actual

#model = 'final_best_concentration_model_withotAudio.keras'


def load_trained_model(model_path='reports/finally_model_without_audio100.keras'):
    """Carga un modelo previamente entrenado"""
    return tf.keras.models.load_model(model_path)

def evaluate_with_trained_model(psd_data, labels, 
                              model_path='reports/finally_model_without_audio100.keras',
                              scaler_path='reports/scaler_without_audio100.save',
                              threshold=0.5):
    # Cargar modelo y scaler
    model = load_trained_model(model_path)
    scaler = joblib.load(scaler_path)  # Carga el scaler original
    
    # Usar transform (NO fit_transform)
    psd_data_scaled = scaler.transform(psd_data)  # <--- Esto es lo crucial
    # Evaluación completa
    y_pred_proba = model.predict(psd_data_scaled)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Métricas
    print("\nMétricas de evaluación:")
    print(classification_report(labels, y_pred, target_names=['Not Focused', 'Focused']))
    
    # Métricas
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    
    print("\nMétricas de evaluación (Dataset Completo):")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    
    
    
    # Matriz de confusión
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Focused', 'Focused'],
                yticklabels=['Not Focused', 'Focused'])
   # plt.title('Matriz de Confusión', pad=20)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    
    return y_pred

# Uso del código:
if __name__ == "__main__":
    # 1. Cargar datos (sin necesidad de dividir para entrenamiento)
    psd_data = pd.read_csv('ProcessedData/Tetris_dataPSD_without_audio.csv', header=None).values
    msc_data = pd.read_csv('ProcessedData/Tetris_DataCoherence_without_audio.csv', header=None).values
    entropy_data = pd.read_csv('ProcessedData/Tetris_dataEntropy_without_audio.csv').values
    
    # 2. Obtener etiquetas (usando tu función automatic_labeling)
    labels, _, _, _ = automatic_labeling_pca(psd_data, msc_data, entropy_data)
    
    # 3. Evaluar con modelo pre-entrenado
    print("\n" + "="*50)
    print("EVALUACIÓN CON MODELO PRE-ENTRENADO")
    print("="*50)
    
    evaluate_with_trained_model(psd_data, labels)