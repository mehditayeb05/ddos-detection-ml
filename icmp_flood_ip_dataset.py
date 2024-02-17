import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Generating normal IP address data
np.random.seed(42)
normal_ips = ['192.168.1.' + str(i) for i in range(1, 90)]

# Generating IP address data during an ICMP Flood attack
attack_ips = ['192.168.1.' + str(i) for i in range(91, 254)]

# Creating a pandas DataFrame
df_normal = pd.DataFrame(data=normal_ips, columns=['source_ip'])
df_attack = pd.DataFrame(data=attack_ips, columns=['source_ip'])
df_normal['label'] = 0  # 0 to indicate normal activity
df_attack['label'] = 1  # 1 to indicate an attack

# Concatenating normal and attack data
df = pd.concat([df_normal, df_attack], ignore_index=True)

# Shuffling the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Splitting the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# Saving the training and testing sets to CSV files
train_data.to_csv('train_icmp_flood_ip_dataset.csv', index=False)
test_data.to_csv('test_icmp_flood_ip_dataset.csv', index=False)
