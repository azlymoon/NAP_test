import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the files
file_1 = 'data/exp53_20240422-161328_gan_adversarial.csv'
file_2 = 'data/exp51_20240417-214543_gan_adversarial.csv'
file_3 = 'data/exp32_20240313-121131_gan_adversarial.csv'

df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)

# Calculate moving averages and standard deviations with a window size of 15 for smoother trends
window_size = 15
df1['Moving Average'] = df1['Value'].rolling(window=window_size).mean()
df1['Std Dev'] = df1['Value'].rolling(window=window_size).std()

df2['Moving Average'] = df2['Value'].rolling(window=window_size).mean()
df2['Std Dev'] = df2['Value'].rolling(window=window_size).std()

df3['Moving Average'] = df3['Value'].rolling(window=window_size).mean()
df3['Std Dev'] = df3['Value'].rolling(window=window_size).std()

# Plotting the moving averages and standard deviations
plt.figure(figsize=(14, 8), dpi=300)  # High resolution for publication

# Plot with Learning Rate information
plt.plot(df1['Step'], df1['Moving Average'], label='patch27 (LR=0.01)', color='C0')
plt.fill_between(df1['Step'], df1['Moving Average'] - df1['Std Dev'], df1['Moving Average'] + df1['Std Dev'], color='C0', alpha=0.2)

plt.plot(df2['Step'], df2['Moving Average'], label='patch25 (LR=0.001)', color='C1')
plt.fill_between(df2['Step'], df2['Moving Average'] - df2['Std Dev'], df2['Moving Average'] + df2['Std Dev'], color='C1', alpha=0.2)

plt.plot(df3['Step'], df3['Moving Average'], label='patch17 (LR=0.005)', color='C2')
plt.fill_between(df3['Step'], df3['Moving Average'] - df3['Std Dev'], df3['Moving Average'] + df3['Std Dev'], color='C2', alpha=0.2)

# plt.title('Moving Average of Detection Loss in Adversarial Patch Training', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Detection loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim(0, 1000)
plt.ylim()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('gan_adversarial_training_detection_loss.png', dpi=300)  # Save the plot with high resolution
