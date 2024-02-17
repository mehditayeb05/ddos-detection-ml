import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the trained model
model = joblib.load('trained_logistic_model_ip.pkl')

# Load the test data with IP addresses (using the same file as the training data)
test_data = pd.read_csv('test_icmp_flood_ip_dataset.csv')

# Load the scaler used during training
scaler = joblib.load('trained_scaler_ip.pkl')

def preprocess_data(data, scaler):
    # Perform necessary preprocessing on the data
    data['source_ip'] = data['source_ip'].apply(lambda ip: int(ip.split('.')[-1]))
    data[['source_ip']] = scaler.transform(data[['source_ip']])
    return data

def detect_attack(data):
    # Apply the model to detect attacks
    predictions = model.predict(data[['source_ip']])
    return predictions

if __name__ == "__main__":
    # Preprocess the data
    test_data = preprocess_data(test_data, scaler)

    # Detect attacks
    predictions = detect_attack(test_data)

    # Display the results or take other necessary actions
    print(predictions)
    results = pd.DataFrame({'Actual': test_data['label'], 'Predicted': predictions})

    # Print part of the DataFrame for further analysis
    print(results.head(50))  

    # Calculate and display the accuracy
    accuracy = accuracy_score(test_data['label'], predictions)
    print(f'Accuracy: {accuracy}')
