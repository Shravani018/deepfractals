# 🌀 deepfractal

Deep learning math iterated across the complex plane. The fractal geometry that emerges reflects how networks actually behave; where they converge, where they explode, and what sits at the edge between the two.

---

## Fractals

### 1. Forward Pass Dynamics
**`forward_pass.py`**

Iterates a single neural layer `z → tanh(w·z + b)` as a complex map,where `w` is the weight, `z` is the input, and `b` is the bias.
`w` and `b` are both derived from the same complex coordinate `c`, where `w = c` and `b = c · 0.3j`.
`tanh` normally squashes real values into (−1, 1), but in the complex plane that bound disappears, which is what makes escape and fractal structure possible in the first place.
Color encodes whether the repeated forward pass diverges or stabilizes, and how fast.

<img src="outputs/forward_pass.png" width="400" height="400"/>
