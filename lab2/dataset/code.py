import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for 14 days with minute-level resolution
start_date = datetime(2025, 3, 1)
dates = [start_date + timedelta(minutes=i) for i in range(14 * 24 * 60)]  # 14 days of data

# Create base dataframe with timestamps
df = pd.DataFrame({'timestamp': dates})

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0 = Monday, 6 = Sunday

# Generate normal network traffic pattern with daily cycles
# Network traffic is typically higher during work hours and lower at night
def generate_base_traffic(hour, day_of_week):
    # Weekday patterns (Mon-Fri)
    if day_of_week < 5:
        if 9 <= hour <= 17:  # Work hours
            return np.random.normal(800, 120)  # Higher mean, moderate variance
        elif 18 <= hour <= 22:  # Evening
            return np.random.normal(500, 80)   # Medium mean, lower variance
        else:  # Night
            return np.random.normal(200, 50)   # Low mean, low variance
    # Weekend patterns (Sat-Sun)
    else:
        if 10 <= hour <= 20:  # Active hours
            return np.random.normal(600, 100)  # Medium-high mean
        else:  # Non-active hours
            return np.random.normal(300, 70)   # Low-medium mean

# Apply the base traffic pattern
df['network_packet_size'] = df.apply(
    lambda row: generate_base_traffic(row['hour'], row['day_of_week']), 
    axis=1
)

# Add some trend and seasonality
df['network_packet_size'] = df['network_packet_size'] + \
                            30 * np.sin(np.linspace(0, 8*np.pi, len(df))) + \
                            15 * np.sin(np.linspace(0, 30*np.pi, len(df)))

# Add random noise
df['network_packet_size'] = df['network_packet_size'] + np.random.normal(0, 25, size=len(df))

# Ensure all values are positive
df['network_packet_size'] = df['network_packet_size'].clip(lower=10)

# Insert anomalies (potential cyber attacks)
attack_periods = [
    # Format: (start_day, start_hour, duration_hours, attack_type)
    (2, 14, 3, 'DDoS'),         # Day 2, 2pm, 3 hours
    (5, 10, 2, 'Port Scan'),    # Day 5, 10am, 2 hours
    (8, 22, 4, 'Data Exfil'),   # Day 8, 10pm, 4 hours
    (11, 3, 2, 'DDoS'),         # Day 11, 3am, 2 hours
    (12, 16, 1, 'Port Scan')    # Day 12, 4pm, 1 hour
]

# Function to apply different attack patterns
def apply_attack(base_value, attack_type):
    if attack_type == 'DDoS':
        # DDoS: very high packet sizes
        return base_value * 3 + np.random.normal(500, 200)
    elif attack_type == 'Port Scan':
        # Port Scan: many small packets
        return np.random.normal(200, 50)
    elif attack_type == 'Data Exfil':
        # Data Exfiltration: moderately high packet sizes at unusual times
        return base_value * 1.5 + np.random.normal(300, 100)
    return base_value

# Mark attack periods and modify network_packet_size
df['is_attack'] = 0
df['attack_type'] = None

for day, hour, duration, attack_type in attack_periods:
    start_idx = day * 24 * 60 + hour * 60
    end_idx = start_idx + duration * 60
    
    # Mark the attack period
    df.loc[start_idx:end_idx, 'is_attack'] = 1
    df.loc[start_idx:end_idx, 'attack_type'] = attack_type
    
    # Modify packet sizes during attack
    for idx in range(start_idx, end_idx + 1):
        if idx < len(df):
            df.loc[idx, 'network_packet_size'] = apply_attack(
                df.loc[idx, 'network_packet_size'], 
                attack_type
            )

# Add a connection count feature
df['connection_count'] = df['network_packet_size'] / np.random.uniform(10, 50, size=len(df))
df['connection_count'] = df['connection_count'].round().astype(int)

# Add protocol distribution
protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
df['protocol'] = np.random.choice(protocols, size=len(df), p=[0.4, 0.3, 0.05, 0.15, 0.1])

# During port scans, change protocol distribution
port_scan_indices = df[df['attack_type'] == 'Port Scan'].index
df.loc[port_scan_indices, 'protocol'] = np.random.choice(
    protocols, size=len(port_scan_indices), p=[0.7, 0.2, 0.1, 0, 0]
)

# Calculate some additional features
df['packet_per_minute'] = df['network_packet_size'] / df['connection_count']
df['packet_per_minute'] = df['packet_per_minute'].clip(lower=1)

# Convert protocols to numeric for machine learning
protocol_map = {proto: i for i, proto in enumerate(protocols)}
df['protocol_code'] = df['protocol'].map(protocol_map)

# Save to CSV
df.to_csv('cybersecurity_intrusion_detection.csv', index=False)
print("Dataset generated and saved to cybersecurity_intrusion_detection.csv")

# Visualize the dataset
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(df['network_packet_size'])
plt.title('Network Packet Size Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Packet Size')
attack_starts = [day * 24 * 60 + hour * 60 for day, hour, _, _ in attack_periods]
for start in attack_starts:
    plt.axvline(x=start, color='r', linestyle='--', alpha=0.7)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(df.index, df['network_packet_size'], c=df['is_attack'], cmap='coolwarm', alpha=0.5)
plt.title('Network Packet Size with Attack Periods Highlighted')
plt.xlabel('Time (minutes)')
plt.ylabel('Packet Size')
plt.colorbar(label='Attack (1) vs Normal (0)')
plt.grid(True)

plt.tight_layout()
plt.savefig('dataset_visualization.png')
plt.show()

print("Generated dataset with shape:", df.shape)
print("\nFeatures in the dataset:")
print(df.columns.tolist())
print("\nSample of the dataset:")
print(df.head())
print("\nStatistics:")
print(df.describe())
print("\nAttack distribution:")
print(df['is_attack'].value_counts())
print("\nAttack types:")
print(df['attack_type'].value_counts(dropna=False))