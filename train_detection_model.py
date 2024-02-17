import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Charger les données d'entraînement avec des adresses IP
train_data = pd.read_csv('train_icmp_flood_ip_dataset.csv')

# Prétraiter les données (vous devez adapter le prétraitement en fonction de vos besoins)
# Ici, on convertit les adresses IP en représentation numérique
train_data['source_ip'] = train_data['source_ip'].apply(lambda ip: int(ip.split('.')[-1]))

# Utiliser le StandardScaler pour normaliser les données (vous pouvez adapter cela)
scaler = StandardScaler()
train_data[['source_ip']] = scaler.fit_transform(train_data[['source_ip']])

# Créer et entraîner le modèle de régression logistique
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(train_data[['source_ip']], train_data['label'])

# Sauvegarder le modèle
joblib.dump(logistic_model, 'trained_logistic_model_ip.pkl')
# Sauvegarder le scaler utilisé lors de l'entraînement
joblib.dump(scaler, 'trained_scaler_ip.pkl')
