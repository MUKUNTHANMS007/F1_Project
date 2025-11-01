from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import tensorflow as tf
from datetime import datetime
import os
import warnings
from keras.saving import register_keras_serializable
from keras.mixed_precision import Policy

@register_keras_serializable()
class DTypePolicy(Policy):
    pass

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Serve frontend HTML on root
@app.route('/')
def index():
    with open('apex_grid_ai.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

def load_models():
    """Load all trained models"""
    print("Loading trained models...")
    dl_model = tf.keras.models.load_model("models/deep_learning_model.h5")
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    print("Models loaded successfully!")
    return {
        'dl_model': dl_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'metrics': metrics
    }

models = load_models()

def preprocess_input(data_dict):
    try:
        df = pd.DataFrame([data_dict])
        for col in ['weather_state', 'sector']:
            if col in df.columns:
                df[col] = models['label_encoders'][col].transform(df[col])
        df = df[models['feature_names']]
        return df
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")

def calculate_strategy(speed, fuel, tire_wear, lap):
    avg_tire_wear_rate = tire_wear / max(lap, 1)
    laps_until_pit = max(1, int((100 - tire_wear) / max(avg_tire_wear_rate, 0.1)))
    if tire_wear < 30:
        tire_compound = "Soft"
    elif tire_wear < 60:
        tire_compound = "Medium"
    else:
        tire_compound = "Hard"
    fuel_per_lap = fuel / max(lap, 1)
    remaining_laps = max(1, int(fuel / max(fuel_per_lap, 0.1)))
    drs_available = True if speed > 150 else False
    return {
        'pit_stop_lap': laps_until_pit,
        'tire_compound': tire_compound,
        'fuel_load_next_stint': round(fuel_per_lap * 20, 2),
        'drs_available': drs_available,
        'remaining_laps': remaining_laps
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Apex Grid AI Backend'
    }), 200

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    return jsonify({
        'status': 'success',
        'models': {
            'deep_learning': {
                'type': 'Neural Network',
                'accuracy': models['metrics'].get('deep_learning', {}).get('r2', 0),
                'rmse': models['metrics'].get('deep_learning', {}).get('rmse', 0)
            },
            'random_forest': {
                'type': 'Random Forest',
                'accuracy': models['metrics'].get('random_forest', {}).get('r2', 0),
                'rmse': models['metrics'].get('random_forest', {}).get('rmse', 0)
            }
        },
        'metrics': models['metrics'],
        'features_count': len(models['feature_names']),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/predict/laptime', methods=['POST'])
def predict_laptime():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        df = preprocess_input(data)
        X_scaled = models['scaler'].transform(df)
        prediction = models['dl_model'](X_scaled)
        prediction = float(prediction.numpy()[0][0])
        confidence = models['metrics']['deep_learning']['r2'] * 100
        return jsonify({
            'status': 'success',
            'prediction': {
                'laptime': round(float(prediction), 3),
                'unit': 'seconds',
                'confidence': round(float(confidence), 2),
                'model': 'deep_learning'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/strategy', methods=['POST'])
def predict_strategy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        df = preprocess_input(data)
        prediction = models['rf_model'].predict(df)[0]
        speed = float(data.get('speed', 0))
        fuel = float(data.get('fuel_load', 0))
        tire_wear = float(data.get('tire_wear_front_left', 0))
        lap = int(data.get('lap', 1))
        strategy = calculate_strategy(speed, fuel, tire_wear, lap)
        confidence = models['metrics']['random_forest']['r2'] * 100
        return jsonify({
            'status': 'success',
            'strategy': strategy,
            'confidence': round(float(confidence), 2),
            'model': 'random_forest',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        df = preprocess_input(data)
        X_scaled = models['scaler'].transform(df)
        dl_pred = models['dl_model'](X_scaled)
        dl_pred = float(dl_pred.numpy()[0][0])
        rf_pred = models['rf_model'].predict(df)[0]
        ensemble_pred = (0.5 * dl_pred + 0.5 * rf_pred)
        avg_confidence = ((models['metrics']['deep_learning']['r2'] +
                           models['metrics']['random_forest']['r2']) / 2) * 100
        return jsonify({
            'status': 'success',
            'prediction': {
                'value': round(float(ensemble_pred), 3),
                'dl_prediction': round(float(dl_pred), 3),
                'rf_prediction': round(float(rf_pred), 3),
                'confidence': round(float(avg_confidence), 2)
            },
            'model': 'ensemble',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze/telemetry', methods=['POST'])
def analyze_telemetry():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        df = preprocess_input(data)
        X_scaled = models['scaler'].transform(df)
        laptime = models['dl_model'](X_scaled)
        laptime = float(laptime.numpy()[0][0])
        strategy_score = models['rf_model'].predict(df)[0]
        speed = float(data.get('speed', 0))
        engine_rpm = float(data.get('engine_rpm', 0))
        tire_wear_avg = (float(data.get('tire_wear_front_left', 0)) +
                        float(data.get('tire_wear_front_right', 0)) +
                        float(data.get('tire_wear_rear_left', 0)) +
                        float(data.get('tire_wear_rear_right', 0))) / 4
        fuel = float(data.get('fuel_load', 0))
        damage = int(data.get('damage_flag', 0))
        health_score = 100 - (tire_wear_avg * 0.5 + damage * 20)
        alerts = []
        if tire_wear_avg > 80:
            alerts.append({'severity': 'high', 'message': 'Critical tire wear - pit stop recommended'})
        if engine_rpm > 13000:
            alerts.append({'severity': 'medium', 'message': 'High RPM - reduce throttle'})
        if fuel < 10:
            alerts.append({'severity': 'high', 'message': 'Low fuel - pit stop required'})
        if damage > 0:
            alerts.append({'severity': 'high', 'message': 'Vehicle damage detected'})
        return jsonify({
            'status': 'success',
            'telemetry_analysis': {
                'speed': speed,
                'engine_rpm': engine_rpm,
                'average_tire_wear': round(tire_wear_avg, 2),
                'fuel_load': fuel,
                'vehicle_health': round(float(health_score), 2),
                'damage_flag': bool(damage),
                'predicted_laptime': round(float(laptime), 3),
                'strategy_score': round(float(strategy_score), 2)
            },
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({
        'status': 'success',
        'features': models['feature_names'],
        'total_features': len(models['feature_names']),
        'categorical_features': ['weather_state', 'sector'],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data_list = request.get_json()
        if not isinstance(data_list, list):
            return jsonify({'error': 'Expected list of data objects'}), 400
        results = []
        for idx, data_dict in enumerate(data_list):
            try:
                df = preprocess_input(data_dict)
                X_scaled = models['scaler'].transform(df)
                laptime = models['dl_model'](X_scaled)
                laptime = float(laptime.numpy()[0][0])
                strategy = models['rf_model'].predict(df)[0]
                results.append({
                    'index': idx,
                    'laptime': round(float(laptime), 3),
                    'strategy_score': round(float(strategy), 2),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e),
                    'status': 'failed'
                })
        return jsonify({
            'status': 'success',
            'predictions': results,
            'total': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    try:
        feature_importance = {}
        for name, importance in zip(models['feature_names'], models['rf_model'].feature_importances_):
            feature_importance[name] = float(importance)
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        top_features = dict(list(sorted_features.items())[:15])
        return jsonify({
            'status': 'success',
            'feature_importance': top_features,
            'total_features': len(feature_importance),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
