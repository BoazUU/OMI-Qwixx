import numpy as np

with np.load("theta_20260930_153222.npz", allow_pickle=True) as data:
    print("Arrays in bestand:", data.files)

    for name in data.files:
        print(f"\n{name}:")
        print(data[name])