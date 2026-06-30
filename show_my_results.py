import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Use the exact folder location we verified outside your directory
folder_path = "../data/rps_40000_False/"

# We will plot the main benchmarks and one of the Boltzmann models to keep the graph clean
files_to_plot = {
    "MLPRegressor": folder_path + "MLPRegressor.pkl",
    "Bayesian": folder_path + "BayesianEstimator.pkl",
    "Boltzmann Model 1": folder_path + "Boltzmann Learning | $\\lambda=1.0e-01, \\beta=1.0e+00$.pkl"
}

fig, ax = plt.subplots(figsize=(12, 6))

for label, data_path in files_to_plot.items():
    if not os.path.exists(data_path):
        print(f"Skipping {label}: File not found at {data_path}")
        continue
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # Extract iteration counts and average cost tracking
    # If 'iters' isn't explicitly saved as a key, default to a 40,000 step array
    if isinstance(data, dict) and 'average_costs' in data:
        average_costs = data['average_costs']
        iters = data.get('iters', np.arange(len(average_costs)))
    elif isinstance(data, list):
        # Fallback if the data is saved as a raw list of costs per iteration
        iters = np.arange(len(data))
        average_costs = np.cumsum(data) / (iters + 1)
    else:
        # Fallback if data is a dictionary but with a different structure
        print(f"Data keys for {label}: {list(data.keys()) if hasattr(data, 'keys') else 'No keys'}")
        continue

    # Plot the rolling average cost line for each algorithm
    ax.plot(iters, average_costs, '-', label=label, linewidth=2)

# Draw a vertical line exactly where the non-stationary pattern shift happened
ax.axvline(x=20000, color='red', linestyle='--', label='Opponent Strategy Shift (t=20k)')

ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlabel('Iteration (t)')
ax.set_ylabel('Running Average Cost')
ax.set_title('Algorithm Performance Under Bandit Feedback (M=40,000)')
ax.legend()

# Save the chart as a viewable picture file
output_image = 'bandit_baseline_results.png'
plt.savefig(output_image, dpi=300)
print(f"SUCCESS! Saved your research chart as '{output_image}' in your root folder!")
