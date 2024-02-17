import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data with IP addresses
train_data = pd.read_csv('train_icmp_flood_ip_dataset.csv')

# Preprocess the data (you need to adjust preprocessing as needed)
# Here, we convert IP addresses into numerical representation
train_data['source_ip'] = train_data['source_ip'].apply(lambda ip: int(ip.split('.')[-1]))

# Use StandardScaler to normalize the data (you can adjust this as needed)
scaler = StandardScaler()
train_data[['source_ip']] = scaler.fit_transform(train_data[['source_ip']])

# Create and train the logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(train_data[['source_ip']], train_data['label'])

# Save the model
joblib.dump(logistic_model, 'trained_logistic_model_ip.pkl')
# Save the scaler used during training
joblib.dump(scaler, 'trained_scaler_ip.pkl')
