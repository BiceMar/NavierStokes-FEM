import pandas as pd
import matplotlib.pyplot as plt

# Set a style

print(plt.style.available)
plt.style.use('seaborn-v0_8-white')
# Load the CSV file
file_path = '../build/lift_coefficient.csv'
data = pd.read_csv(file_path)


# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['LiftCoefficient'], marker='o', linestyle='-', color='deepskyblue', linewidth=2, markersize=8, label='Lift Coefficient')
plt.title('Lift Coefficient over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Lift Coefficient', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
# Save the plot to a file instead of showing it
output_path = '../imgs/'
plt.savefig(output_path)
plt.close()
