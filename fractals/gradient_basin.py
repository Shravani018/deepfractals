"""
Fractal 2: Gradient Descent Basin Fractal

For each point in the complex plane, w0 defines a starting weight for gradient descent.
We minimize a small neural loss function L(w) = sum(tanh(w*x) - y)^2 and track two things:
1. Basin: which local minimum gradient descent converges to
2. Speed: how many steps it takes to get there

The fractal boundary between basins is where weight initialization is maximally sensitive,
the exact geometry of why initialization matters in deep learning.
"""
# Importing necessary libraries
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
# Parameters
img_w, img_h=1200, 1200 # Image width and height
max_steps=150 # Maximum gradient descent steps
lr=0.04 # Learning rate
bounds=(-2.5, 2.5, -2.5, 2.5)# (xmin, xmax, ymin, ymax)
# Creating an output directory
os.makedirs("./outputs", exist_ok=True)
# Color palette
palette=LinearSegmentedColormap.from_list("", [
    "#000000","#000820","#001845","#003366",
    "#006699","#00aacc","#00ddaa",
    "#66ff88","#ddffcc","#ffffff"
], N=4096)
# Tiny synthetic dataset: input/output pairs the network tries to fit
xs=np.array([ 0.5, -0.5,  1.0, -1.0,  0.3, -0.8])
ys=np.array([ 0.8, -0.8,  0.6, -0.6,  0.4, -0.5])
# Creating the grid
x0, x1, y0, y1=bounds
# Each point w0 in the complex plane is a starting weight for gradient descent
w=(np.linspace(x0, x1, img_w)[np.newaxis,:] +
   1j*np.linspace(y0, y1, img_h)[:,np.newaxis])
# Initialize arrays for convergence tracking, step count, lyapunov exponent, and done mask
conv_to=np.zeros((img_h, img_w), dtype=complex) # which minimum we converged to
steps_to=np.full((img_h, img_w), max_steps, dtype=float) # how many steps it took
lyapunov=np.zeros((img_h, img_w), dtype=float) # gradient chaos estimate
done=np.zeros((img_h, img_w), dtype=bool)
# Running gradient descent from each starting weight and tracking convergence
for step in range(max_steps):
    mask=~done
    if not np.any(mask): break
    wm=w[mask]
    grad=np.zeros_like(wm)
    # Accumulate gradient over all data points: dL/dw = 2*(tanh(w*x) - y)*(1-tanh^2(w*x))*x
    for xi, yi in zip(xs, ys):
        t=np.tanh(wm * xi)
        grad += 2*(t - yi)*(1 - t**2)*xi
    # Lyapunov: log of gradient magnitude tracks chaos at basin boundaries
    lyapunov[mask] += np.log(np.maximum(np.abs(grad), 1e-15)) / max_steps
    wm_new=wm - lr*grad
    wm_new=np.where(np.isfinite(wm_new), wm_new, wm)
    # Mark converged points: weight barely moved, we found a minimum
    new_done=np.abs(wm_new - wm) < 1e-5
    ri=np.where(mask)
    rc=(ri[0][new_done], ri[1][new_done])
    conv_to[rc]=wm_new[new_done]
    steps_to[rc]=step
    done[rc]=True
    w[mask]=wm_new
# Non-converged points use their final weight as the basin label
conv_to[~done]=w[~done]
# Mapping basin angle, convergence speed, and lyapunov to color
basin_ang=(np.angle(conv_to) + np.pi) / (2*np.pi) # which basin, encoded as angle
speed=1.0 - steps_to / max_steps # fast convergence = bright
ly_n=np.clip((lyapunov-lyapunov.min())/(lyapunov.max()-lyapunov.min()+1e-10), 0, 1)
img=speed**0.55 * (0.5 + 0.5*np.sin(basin_ang*np.pi*10))
img += 0.25*ly_n
img=np.clip(img, 0, 1)
glow=gaussian_filter(img, sigma=2.5)
img=np.clip(img + 0.3*glow, 0, 1)
y_, x_=np.ogrid[:img_h, :img_w]
vig=np.clip(1 - 0.45*((((x_-img_w/2)/(img_w/2))**2+((y_-img_h/2)/(img_h/2))**2)), 0, 1)
img*=vig
# Plotting the fractal
fig, ax=plt.subplots(figsize=(10, 10), facecolor="#000000")
ax.imshow(img, cmap=palette, origin="lower", interpolation="lanczos",
          vmin=0, vmax=1, extent=[x0, x1, y0, y1])
ax.set_title("Gradient Descent Basins: Loss landscape attraction in weight space",
             color="white", fontsize=13, fontweight="bold", fontfamily="monospace", pad=14)
ax.set_xlabel("Re(w0)", color="#446688", fontsize=9, fontfamily="monospace")
ax.set_ylabel("Im(w0)", color="#446688", fontsize=9, fontfamily="monospace")
ax.tick_params(colors="#334455", labelsize=7)
for s in ax.spines.values(): s.set_edgecolor("#112233")
fig.tight_layout()
plt.savefig("./outputs/gradient_basins.png",
            dpi=180, bbox_inches="tight", facecolor="black")
print("Fractal image saved to ./outputs/gradient_basins.png")