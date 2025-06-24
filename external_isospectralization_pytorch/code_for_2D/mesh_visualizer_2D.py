import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Load your mesh files
V = np.loadtxt("mesh.vert")            # shape (N,3), but Z is all zeros
T = np.loadtxt("mesh.triv", dtype=int) - 1  # shape (M,3), convert to 0-based

# Extract X and Y
X = V[:,0]
Y = V[:,1]

# Build a Triangulation object
triang = Triangulation(X, Y, T)

# Plot
plt.figure(figsize=(6,6))
plt.triplot(triang, color="gray", linewidth=0.5)
plt.scatter(X, Y, s=10, color="red")
plt.gca().set_aspect("equal")
plt.title("Mesh Visualization")
plt.show()
