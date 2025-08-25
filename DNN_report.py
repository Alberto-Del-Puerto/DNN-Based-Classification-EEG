from automatic_labeling02 import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

def load_and_prepare_data():
    psd_data = pd.read_csv('ProcessedData/Tetris_dataPSD_without_audio.csv', header=None).values
    msc_data = pd.read_csv('ProcessedData/Tetris_DataCoherence_without_audio.csv', header=None).values
    entropy_data = pd.read_csv('ProcessedData/Tetris_dataEntropy_without_audio.csv').values
    
    print('Dimensiones de los datos:')
    print(f'PSD: {psd_data.shape}, MSC: {msc_data.shape}, Entropía: {entropy_data.shape}')
    
    labels, _, _, _ = automatic_labeling_pca(psd_data, msc_data, entropy_data)
    
    return psd_data, labels

def create_enhanced_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', 
              input_shape=(input_shape,),
              kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.005)),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')]
    )
    return model


def full_dataset_evaluation(model, psd_data, labels, scaler, threshold=0.5):
   
    psd_data_scaled = scaler.transform(psd_data)
    
 
    y_pred_proba = model.predict(psd_data_scaled)
    y_pred = (y_pred_proba > threshold).astype(int)
    
  
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    
    print("\nMétricas de evaluación (Dataset Completo):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
   
    report = classification_report(labels, y_pred, target_names=['Not Focused', 'Focused'])
    return y_pred, accuracy, precision, recall, f1, report

def main():
    
    psd_data, labels = load_and_prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        psd_data, labels, test_size=0.2, random_state=42, stratify=labels)
    
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
   
    best_model = None
    best_accuracy_score = 0
    best_f1_score = 0
    j=0
    metrics_history = []
    classification_reports = []  
    
    with open("reports/finally_reports_without_audio100.txt", "w") as f:
        for i in range(10):
            print(f"\nEjecución {i+1}")
            
            # Crear modelo
            model = create_enhanced_model(X_train_scaled.shape[1])
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_recall', patience=15, mode='max', verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
                ModelCheckpoint('best_model.keras', monitor='val_recall', 
                                save_best_only=True, mode='max')
            ]
            
          
            model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                callbacks=callbacks,
                class_weight={0: 1, 1: 2},  
                verbose=1
            )
            
          
            best_model_current = tf.keras.models.load_model('best_model.keras')
            
            print("\n" + "="*50)
            print("EVALUACIÓN CON TODO EL DATASET PSD")
            print("="*50)
            
            y_pred, accuracy, precision, recall, f1, report = full_dataset_evaluation(best_model_current, psd_data, labels, scaler)
            
          
            metrics_history.append({
                'Ejecución': i+1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            f.write(f"\nEjecución {i+1}:\n")
            f.write(report)
            f.write("\n" + "="*50 + "\n")
            
            # f.write("\nMétricas de evaluación (Dataset Completo):\n")
            # f.write(f"Accuracy: {accuracy:.4f}\n")
            # f.write(f"Precision: {precision:.4f}\n")
            # f.write(f"Recall: {recall:.4f}\n")
            # f.write(f"F1-Score: {f1:.4f}\n")
            
            
            
            # Comparar y almacenar el mejor modelo basado en el F1-Score
            # if f1 > best_f1_score:
            #     best_f1_score = f1
            #     best_model = best_model_current
            if accuracy > best_accuracy_score:
                best_accuracy_score = accuracy
                best_model = best_model_current
                j = i+1
    
   
    best_model.save('reports/finally_model_without_audio100.keras')
    print("\nEl mejor modelo ha sido guardado como 'best_model.keras'")
    joblib.dump(scaler, 'reports/scaler_without_audio100.save')  # <-- Añade esta línea

    print("el mejor modelo es: ", j)
    print("accuracy:", best_accuracy_score)

    metrics_df = pd.DataFrame(metrics_history)
    print("\nHistorial de métricas:")
    print(metrics_df)

if __name__ == "__main__":
    main()
