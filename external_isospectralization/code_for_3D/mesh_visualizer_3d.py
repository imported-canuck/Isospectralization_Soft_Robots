import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import but required for 3D projection

# --- NEW: file picker UI (minimal change) ---
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()  # hide the empty tkinter window
root.attributes("-topmost", True)  # bring dialog to front (optional, but nice)

vert_path = askopenfilename(
    title="Select mesh.vert file",
    filetypes=[("VERT files", "*.vert"), ("All files", "*.*")]
)
triv_path = askopenfilename(
    title="Select mesh.triv file",
    filetypes=[("TRIV files", "*.triv"), ("All files", "*.*")]
)

root.destroy()

if not vert_path or not triv_path:
    raise SystemExit("File selection cancelled. Please select both mesh.vert and mesh.triv.")
# --- END NEW ---

# Load mesh files
V = np.loadtxt(vert_path)                # expects lines: X Y Z
T = np.loadtxt(triv_path, dtype=int) - 1 # expects 1-based indices

# Separate coordinates
X, Y, Z = V.T

# Create figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the triangular surface
ax.plot_trisurf(X, Y, Z, triangles=T, linewidth=0.2, antialiased=True,
                color='lightgray', edgecolor='k', alpha=0.8)

# Overlay the mesh vertices
ax.scatter(X, Y, Z, s=8, color='red')

# Function to set equal aspect ratio on 3D axes
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d(x_middle - plot_radius, x_middle + plot_radius)
    ax.set_ylim3d(y_middle - plot_radius, y_middle + plot_radius)
    ax.set_zlim3d(z_middle - plot_radius, z_middle + plot_radius)

set_axes_equal(ax)

# Make x,y,z use the same on-screen scale (Matplotlib >= 3.3)
ax.set_box_aspect((1, 1, 1))

# Optional: remove perspective distortion (often helps spheres look right)
ax.set_proj_type('ortho')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Mesh Visualization")
plt.tight_layout()
plt.show()
