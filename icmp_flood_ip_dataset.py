import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Génération de données d'adresses IP normales
np.random.seed(42)
normal_ips = ['192.168.1.' + str(i) for i in range(1, 90)]

# Génération de données d'adresses IP lors d'une attaque ICMP Flood
attack_ips = ['192.168.1.' + str(i) for i in range(91, 254)]

# Création d'un DataFrame pandas
df_normal = pd.DataFrame(data=normal_ips, columns=['source_ip'])
df_attack = pd.DataFrame(data=attack_ips, columns=['source_ip'])
df_normal['label'] = 0  # 0 pour indiquer une activité normale
df_attack['label'] = 1  # 1 pour indiquer une attaque

# Concaténation des données normales et d'attaque
df = pd.concat([df_normal, df_attack], ignore_index=True)

# Mélange des données
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Division des données en ensemble d'entraînement et ensemble de test
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# Sauvegarde des ensembles d'entraînement et de test en fichiers CSV
train_data.to_csv('train_icmp_flood_ip_dataset.csv', index=False)
test_data.to_csv('test_icmp_flood_ip_dataset.csv', index=False)
