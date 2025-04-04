# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
packet_size_scaled = scaler.fit_transform(packet_size_data.reshape(-1, 1))

# Define sequence length (window size)
time_steps = 60
sequece_len = len(packet_size_scaled) - time_steps

