import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Charger le modèle entraîné
model = joblib.load('trained_logistic_model_ip.pkl')

# Charger les données de test avec des adresses IP (utilisation du même fichier que les données d'entraînement)
test_data = pd.read_csv('test_icmp_flood_ip_dataset.csv')

# Charger le scaler utilisé lors de l'entraînement
scaler = joblib.load('trained_scaler_ip.pkl')

def preprocess_data(data, scaler):
    # Effectuer le prétraitement nécessaire sur les données
    data['source_ip'] = data['source_ip'].apply(lambda ip: int(ip.split('.')[-1]))
    data[['source_ip']] = scaler.transform(data[['source_ip']])
    return data

def detect_attack(data):
    # Appliquer le modèle pour détecter les attaques
    predictions = model.predict(data[['source_ip']])
    return predictions

if __name__ == "__main__":
    # Prétraiter les données
    test_data = preprocess_data(test_data, scaler)

    # Détecter les attaques
    predictions = detect_attack(test_data)

    # Afficher les résultats ou prendre d'autres mesures nécessaires
    print(predictions)
    results = pd.DataFrame({'Actual': test_data['label'], 'Predicted': predictions})

    # Imprimer une partie du DataFrame pour une analyse plus détaillée
    print(results.head(50))  

    # Calculer et afficher l'accuracy
    accuracy = accuracy_score(test_data['label'], predictions)
    print(f'Accuracy: {accuracy}')
