import requests
import time
import random

URL = "http://localhost:5000/predict"

# Contoh data dummy
def generate_dummy_data():
    return {
        "features": [
            random.uniform(0, 14),      # ph
            random.uniform(100, 300),   # Hardness
            random.uniform(10000, 50000), # Solids
            random.uniform(4, 10),      # Chloramines
            random.uniform(200, 500),   # Sulfate
            random.uniform(300, 600),   # Conductivity
            random.uniform(10, 20),     # Organic_carbon
            random.uniform(50, 100),    # Trihalomethanes
            random.uniform(2, 5)        # Turbidity
        ]
    }

print("Memulai simulasi traffic ke model...")

success = 0
fails = 0

while True:
    try:
        payload = generate_dummy_data()
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print(f"Request OK | Pred: {response.json()['prediction']}")
            success += 1
        else:
            print(f"Request Failed: {response.text}")
            fails += 1
            
        time.sleep(random.uniform(0.5, 2.0))
        
    except Exception as e:
        print(f"Error connection: {e}")
        time.sleep(5)