import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir="app/models", metadata_dir="app/models/metadata"):
        self.models_dir = models_dir
        self.metadata_dir = metadata_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
    
    def train_and_save(self, X, y, model_name):
        """Entrenar y guardar modelo con metadatos"""
        try:
            # Convertir a numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Verificar dimensiones
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
            
            # Verificar que haya al menos 2 clases
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise ValueError("Se necesitan al menos 2 clases diferentes para entrenar")
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Guardar modelo
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Guardar metadatos
            metadata = {
                'accuracy': float(accuracy),
                'n_samples': len(X),
                'n_samples_per_class': dict(Counter(y)),
                'classes': unique_classes.tolist(),
                'classification_report': report,
                'feature_importance': model.feature_importances_.tolist()
            }
            
            metadata_path = os.path.join(self.metadata_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modelo '{model_name}' guardado. Precisión: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'n_samples': len(X),
                'classification_report': report,
                'classes': unique_classes.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            raise
    
    def list_models(self):
        """Listar todos los modelos disponibles"""
        try:
            models = []
            for file in os.listdir(self.models_dir):
                if file.endswith('.joblib'):
                    model_name = file.replace('.joblib', '')
                    models.append(model_name)
            return sorted(models)
        except Exception as e:
            logger.error(f"Error listando modelos: {str(e)}")
            return []
    
    def get_model_info(self, model_name):
        """Obtener información de un modelo específico"""
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{model_name}_metadata.json")
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {str(e)}")
            return None
    
    def predict_with_confidence(self, landmarks, model_name):
        """Predecir con nivel de confianza"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                raise ValueError(f"Modelo '{model_name}' no encontrado")
            
            # Cargar modelo
            model = joblib.load(model_path)
            
            # Convertir landmarks a array 2D
            X = np.array(landmarks).reshape(1, -1)
            
            # Predecir probabilidades
            probabilities = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]
            
            # Obtener confianza y todas las predicciones
            confidence = np.max(probabilities)
            classes = model.classes_
            
            # Crear lista de todas las predicciones ordenadas por confianza
            all_predictions = []
            for i, prob in enumerate(probabilities):
                all_predictions.append({
                    'class': classes[i],
                    'confidence': float(prob)
                })
            
            # Ordenar por confianza descendente
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise
    
    def delete_model(self, model_name):
        """Eliminar modelo y sus metadatos"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            metadata_path = os.path.join(self.metadata_dir, f"{model_name}_metadata.json")
            
            deleted_files = []
            
            if os.path.exists(model_path):
                os.remove(model_path)
                deleted_files.append('modelo')
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                deleted_files.append('metadatos')
            
            return deleted_files
            
        except Exception as e:
            logger.error(f"Error eliminando modelo: {str(e)}")
            raise