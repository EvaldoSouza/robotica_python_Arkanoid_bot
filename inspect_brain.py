import pickle
import numpy as np
import sys

# Change this path to point to the specific session file you want to inspect
FILE_PATH = "arkanoid_best_brain.pkl" 

with open(FILE_PATH, "rb") as f:
    payload = pickle.load(f)

# Set NumPy to print the whole array without truncating it (optional)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

print(f"--- Brain Data ---")
print(f"Exploration Rate (Epsilon): {payload['epsilon']:.4f}")
print(f"Best Survival Frames: {payload['best_survival']}")
print("\n--- Q-Table ---")
print(payload['q_table'])