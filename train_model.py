import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================================================ 
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class DataProcessor:
    """Handle all data loading and preprocessing"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess(self):
        """Preprocess the dataset"""
        print("Starting data preprocessing...")
        
        data = self.df.copy()
        self.categorical_features = ['weather_state', 'sector']
        
        for col in self.categorical_features:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        
        self.feature_names = data.columns.tolist()
        
        data['lap_time_target'] = (data['speed'].max() - data['speed']) / 100 + np.random.random(len(data)) * 0.5
        data['pit_stop_target'] = data['pit_stop_flag'].values
        data['tire_wear_target'] = (data['tire_wear_front_left'] + data['tire_wear_front_right'] + 
                                     data['tire_wear_rear_left'] + data['tire_wear_rear_right']) / 4
        data['fuel_target'] = data['fuel_load'].values
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(data)}")
        print(f"Data shape: {data.shape}")
        
        return data
    
    def get_features_and_targets(self, data):
        """Extract features and targets"""
        target_cols = ['lap_time_target', 'pit_stop_target', 'tire_wear_target', 'fuel_target']
        X = data.drop(columns=target_cols + ['lap', 'time'])
        
        y_lap_time = data['lap_time_target'].values
        y_pit_stop = data['pit_stop_target'].values
        y_tire_wear = data['tire_wear_target'].values
        y_fuel = data['fuel_target'].values
        
        return X, y_lap_time, y_pit_stop, y_tire_wear, y_fuel
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled


# ============================================================================ 
# 2. DEEP LEARNING MODEL
# ============================================================================

class DeepLearningModel:
    """Deep Learning model for F1 telemetry prediction"""
    
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.history = None
        
    def _build_model(self, input_shape):
        """Build neural network architecture"""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        print("\nTraining Deep Learning Model...")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
        print(f"Model saved to {path}")


# ============================================================================ 
# 3. RANDOM FOREST MODEL
# ============================================================================

class RandomForestModel:
    """Random Forest model for F1 strategy prediction"""
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.feature_importance = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nTraining Random Forest Model...")
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        print("Random Forest training completed!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance_dict = {}
        for name, importance in zip(feature_names, self.feature_importance):
            importance_dict[name] = float(importance)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")


# ============================================================================ 
# 4. MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models(csv_path='F1-synthetic_balanced_V4-1.csv'):
    """Complete training pipeline"""
    
    print("="*80)
    print("APEX GRID AI - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and preprocess data
    processor = DataProcessor(csv_path)
    data = processor.preprocess()
    X, y_lap_time, y_pit_stop, y_tire_wear, y_fuel = processor.get_features_and_targets(data)
    
    # Step 2: Train/Test split
    X_train, X_test, y_lap_train, y_lap_test = train_test_split(
        X, y_lap_time, test_size=0.2, random_state=42
    )
    X_train, X_val, y_lap_train, y_lap_val = train_test_split(
        X_train, y_lap_train, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 3: Scale features
    X_train_scaled = processor.scale_features(X_train, fit=True)
    X_val_scaled = processor.scale_features(X_val, fit=False)
    X_test_scaled = processor.scale_features(X_test, fit=False)
    
    # Step 4: Train Deep Learning Model
    print("\n" + "="*80)
    print("TRAINING DEEP LEARNING MODEL")
    print("="*80)
    dl_model = DeepLearningModel(input_shape=X_train_scaled.shape[1])
    dl_model.train(X_train_scaled, y_lap_train, X_val_scaled, y_lap_val, 
                   epochs=50, batch_size=32)
    
    dl_metrics = dl_model.evaluate(X_test_scaled, y_lap_test)
    print("\nDeep Learning Metrics:")
    for metric, value in dl_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Step 5: Train Random Forest Model
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*80)
    rf_model = RandomForestModel(n_estimators=100, max_depth=20)
    rf_model.train(X_train, y_lap_train)
    
    rf_metrics = rf_model.evaluate(X_test, y_lap_test)
    print("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Step 6: Feature Importance
    print("\n" + "="*80)
    print("TOP 10 IMPORTANT FEATURES (Random Forest)")
    print("="*80)
    feature_importance = rf_model.get_feature_importance(X.columns.tolist())
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"{i:2d}. {feature:30s} - {importance:.4f}")
    
    # Step 7: Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    dl_model.save('models/deep_learning_model.h5')
    rf_model.save('models/random_forest_model.pkl')
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(processor.scaler, f)
    
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(processor.label_encoders, f)
    
    with open('models/feature_names.json', 'w') as f:
        json.dump(X.columns.tolist(), f)
    
    metrics_summary = {
        'deep_learning': dl_metrics,
        'random_forest': rf_metrics,
        'feature_importance': feature_importance,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(X.columns)
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\nModels saved successfully!")
    
    return {
        'dl_model': dl_model,
        'rf_model': rf_model,
        'scaler': processor.scaler,
        'label_encoders': processor.label_encoders,
        'feature_names': X.columns.tolist(),
        'metrics': metrics_summary
    }


if __name__ == "__main__":
    results = train_all_models('F1-synthetic_balanced_V4-1.csv')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nModels saved in 'models/' directory:")
    print("  - deep_learning_model.h5")
    print("  - random_forest_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_names.json")
    print("  - metrics.json")
