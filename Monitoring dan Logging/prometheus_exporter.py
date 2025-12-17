from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import pandas as pd
import time
import psutil
import os
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

REQUEST_COUNT = Counter('app_request_count', 'Total Request yang masuk')
PREDICTION_COUNT = Counter('app_prediction_total', 'Total Prediksi per Class', ['prediction_class']) 
LATENCY = Histogram('app_latency_seconds', 'Waktu proses request')
CONFIDENCE_SCORE = Gauge('model_confidence_score', 'Rata-rata confidence score')
INPUT_PH_MEAN = Gauge('input_feature_ph_mean', 'Rata-rata nilai pH input')
INPUT_SULFATE_MEAN = Gauge('input_feature_sulfate_mean', 'Rata-rata nilai Sulfate input')

CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage Percent')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Memory Usage Bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk Usage Percent')
NETWORK_SENT = Gauge('system_network_sent_bytes', 'Bytes Sent')
NETWORK_RECV = Gauge('system_network_recv_bytes', 'Bytes Received')

print("Training Model sebentar...")
data_paths = [
    '../data/water_potability_processed.csv',
    '../Membangun_model/water_potability_processed.csv',
    'data/water_potability_processed.csv'
]
df = None
for path in data_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Data loaded from {path}")
        break

if df is None:
    print("Dataset tidak ketemu, pakai Dummy Data buat inisialisasi.")
    df = pd.DataFrame({
        'ph': [7.0]*10, 'Hardness': [200]*10, 'Solids': [20000]*10, 
        'Chloramines': [7.0]*10, 'Sulfate': [300]*10, 'Conductivity': [400]*10,
        'Organic_carbon': [15.0]*10, 'Trihalomethanes': [60.0]*10, 
        'Turbidity': [4.0]*10, 'Potability': [0]*5 + [1]*5
    })

X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
print("Model Ready!")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    data = request.json
    try:
        features = data['features']
        
        # Update Metrik Fitur
        INPUT_PH_MEAN.set(features[0])
        INPUT_SULFATE_MEAN.set(features[4])
        
        # Prediksi
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0].max()
        
        PREDICTION_COUNT.labels(prediction_class=str(prediction)).inc()
        
        CONFIDENCE_SCORE.set(proba)
        
        duration = time.time() - start_time
        LATENCY.observe(duration)
        
        return jsonify({
            'prediction': int(prediction), 
            'confidence': float(proba),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

def update_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().used)
            DISK_USAGE.set(psutil.disk_usage('/').percent)
            net = psutil.net_io_counters()
            NETWORK_SENT.set(net.bytes_sent)
            NETWORK_RECV.set(net.bytes_recv)
        except Exception:
            pass
        time.sleep(5)

if __name__ == '__main__':
    threading.Thread(target=update_system_metrics, daemon=True).start()
    print("Metrics Server running on port 8000")
    start_http_server(8000)
    print("API Server running on port 5000")
    app.run(host='0.0.0.0', port=5000)